"""Load the Olist CSVs into DuckDB or Postgres and run SQL against either.

DuckDB is the primary engine (no server, one file powers notebooks and the
app). Postgres is supported so CI can prove the SQL in ``sql/`` is portable.
"""

from __future__ import annotations

import decimal
import io
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

import duckdb
import pandas as pd

from olist.config import RAW_DIR, WAREHOUSE_PATH
from olist.schema import TABLES, Table

_PANDAS_TYPES = {
    "TEXT": "string",
    "INTEGER": "Int64",
    "DOUBLE PRECISION": "float64",
}


def read_csv(table: Table, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Read one raw CSV with explicit types (zip prefixes stay text)."""
    path = Path(raw_dir) / table.csv
    dtypes = {c: _PANDAS_TYPES[t] for c, t in table.columns if t in _PANDAS_TYPES}
    dates = [c for c, t in table.columns if t == "TIMESTAMP"]
    df = pd.read_csv(path, dtype=dtypes, encoding="utf-8-sig", keep_default_na=True)
    missing = set(table.column_names) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns {sorted(missing)}")
    for c in dates:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df[table.column_names]


class Engine(Protocol):
    name: str

    def query(self, sql: str) -> pd.DataFrame: ...

    def close(self) -> None: ...


class DuckDBEngine:
    name = "duckdb"

    def __init__(self, path: str | Path = WAREHOUSE_PATH, read_only: bool = True):
        self.con = duckdb.connect(str(path), read_only=read_only)

    def query(self, sql: str) -> pd.DataFrame:
        return self.con.execute(sql).df()

    def close(self) -> None:
        self.con.close()


class PostgresEngine:
    name = "postgres"

    def __init__(self, dsn: str):
        import psycopg

        self.con = psycopg.connect(dsn, autocommit=True)

    def query(self, sql: str) -> pd.DataFrame:
        with self.con.cursor() as cur:
            cur.execute(sql)
            cols = [d.name for d in cur.description]
            rows = cur.fetchall()
        df = pd.DataFrame(rows, columns=cols)
        # Postgres returns NUMERIC (Decimal) for AVG/EXTRACT on integers.
        for c in df.columns:
            if df[c].map(lambda v: isinstance(v, decimal.Decimal)).any():
                df[c] = pd.to_numeric(df[c].map(lambda v: None if v is None else float(v)))
        return df

    def close(self) -> None:
        self.con.close()


def build_duckdb(
    db_path: str | Path = WAREHOUSE_PATH,
    raw_dir: Path = RAW_DIR,
    tables: Iterable[Table] = TABLES,
) -> Path:
    """(Re)create the DuckDB warehouse from the raw CSVs."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    con = duckdb.connect(str(db_path))
    try:
        for table in tables:
            df = read_csv(table, raw_dir)
            con.execute(table.ddl())
            con.register("_df", df)
            con.execute(f"INSERT INTO {table.name} SELECT * FROM _df")
            con.unregister("_df")
    finally:
        con.close()
    return db_path


def build_postgres(dsn: str, raw_dir: Path = RAW_DIR, tables: Iterable[Table] = TABLES) -> None:
    """(Re)create the same tables in Postgres using COPY."""
    import psycopg

    with psycopg.connect(dsn, autocommit=True) as con:
        for table in tables:
            df = read_csv(table, raw_dir)
            with con.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {table.name}")
                cur.execute(table.ddl())
                buf = io.StringIO()
                df.to_csv(buf, index=False, header=False, date_format="%Y-%m-%d %H:%M:%S")
                buf.seek(0)
                cols = ", ".join(table.column_names)
                with cur.copy(f"COPY {table.name} ({cols}) FROM STDIN WITH (FORMAT csv)") as cp:
                    while chunk := buf.read(1 << 20):
                        cp.write(chunk)
                cur.execute(f"ANALYZE {table.name}")


def connect(engine: str = "duckdb", **kwargs) -> Engine:
    if engine == "duckdb":
        return DuckDBEngine(kwargs.get("path", WAREHOUSE_PATH))
    if engine == "postgres":
        dsn = kwargs.get("dsn") or os.environ.get("OLIST_PG_DSN")
        if not dsn:
            raise ValueError("Set OLIST_PG_DSN to use the postgres engine")
        return PostgresEngine(dsn)
    raise ValueError(f"unknown engine {engine!r}")

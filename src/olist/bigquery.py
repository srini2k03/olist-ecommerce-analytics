"""Run the analytics layer on Google BigQuery (sandbox) and check it against DuckDB.

The marts are built once by the DuckDB pipeline, exported to Parquet and loaded into a
BigQuery dataset (load jobs are free, and the BigQuery sandbox needs no billing account).
Each analytics query is then translated to BigQuery SQL:

* most queries are translated automatically with sqlglot (DuckDB -> BigQuery), plus two
  fixes sqlglot leaves alone: ``EXTRACT(DOW ...)`` (0 = Sunday) becomes
  ``EXTRACT(DAYOFWEEK ...) - 1`` (BigQuery counts 1 = Sunday), and ``DATE - DATE``
  becomes ``DATE_DIFF(a, b, DAY)``;
* queries using ``PERCENTILE_CONT ... WITHIN GROUP`` have a hand-written BigQuery version
  in ``sql/bigquery/<name>.sql``, because BigQuery's exact percentile only exists as an
  analytic function.

``google-cloud-bigquery`` is imported lazily so the translator works (and is tested)
without Google Cloud libraries or credentials.
"""

from __future__ import annotations

import decimal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import sqlglot
from sqlglot import exp

from olist.config import SQL_DIR

if TYPE_CHECKING:  # pragma: no cover
    from olist.db import DuckDBEngine
    from olist.sql_runner import Query

OVERRIDE_DIR = SQL_DIR / "bigquery"
MART_TABLES = (
    "fct_orders",
    "fct_order_items",
    "fct_payments",
    "fct_reviews",
    "dim_customers",
    "dim_products",
    "dim_sellers",
)


def _is_date_cast(node: exp.Expression) -> bool:
    return isinstance(node, exp.Cast) and node.to.is_type(exp.DataType.Type.DATE)


def _bigquery_fixes(node: exp.Expression) -> exp.Expression:
    if isinstance(node, exp.Extract) and node.this.name.upper() == "DOW":
        dayofweek = exp.Extract(this=exp.var("DAYOFWEEK"), expression=node.expression)
        return exp.paren(exp.Sub(this=dayofweek, expression=exp.Literal.number(1)))
    if isinstance(node, exp.Sub) and _is_date_cast(node.this) and _is_date_cast(node.expression):
        return exp.DateDiff(this=node.this, expression=node.expression, unit=exp.var("DAY"))
    return node


def translate(sql: str) -> str:
    """DuckDB SQL -> BigQuery SQL (sqlglot plus the DOW and DATE - DATE fixes)."""
    tree = sqlglot.parse_one(sql, read="duckdb")
    return tree.transform(_bigquery_fixes).sql(dialect="bigquery", pretty=True)


def override_path(name: str, override_dir: Path = OVERRIDE_DIR) -> Path:
    return Path(override_dir) / f"{name}.sql"


def bigquery_sql(query: Query, override_dir: Path = OVERRIDE_DIR) -> tuple[str, str]:
    """The BigQuery text for an analytics query and how it was produced."""
    path = override_path(query.name, override_dir)
    if path.exists():
        return path.read_text(encoding="utf-8"), "hand-written"
    return translate(query.sql), "translated"


def _to_plain(df: pd.DataFrame) -> pd.DataFrame:
    """BigQuery dtypes -> the plain types the cross-engine comparison expects."""
    out = df.copy()
    for c in out.columns:
        s = out[c]
        name = str(s.dtype)
        if name == "dbdate":
            out[c] = s.map(lambda v: None if pd.isna(v) else v.isoformat())
        elif s.dtype == object and s.map(lambda v: isinstance(v, decimal.Decimal)).any():
            out[c] = s.map(lambda v: None if v is None else float(v)).astype(float)
    return out


@dataclass
class JobStats:
    bytes_processed: int = 0
    elapsed_ms: float = 0.0
    cache_hit: bool = False


@dataclass
class BigQueryEngine:
    """Minimal engine with the same ``query`` interface as the DuckDB and Postgres engines."""

    project: str
    dataset: str = "olist"
    location: str = "US"
    name: str = "bigquery"
    last_job: JobStats = field(default_factory=JobStats)
    _client: Any = None

    @property
    def client(self) -> Any:
        if self._client is None:
            from google.cloud import bigquery

            self._client = bigquery.Client(project=self.project, location=self.location)
        return self._client

    def query(self, sql: str) -> pd.DataFrame:
        from google.cloud import bigquery

        config = bigquery.QueryJobConfig(
            default_dataset=f"{self.project}.{self.dataset}", use_query_cache=False
        )
        start = time.perf_counter()
        job = self.client.query(sql, job_config=config)
        df = job.result().to_dataframe()
        self.last_job = JobStats(
            bytes_processed=int(job.total_bytes_processed or 0),
            elapsed_ms=(time.perf_counter() - start) * 1000,
            cache_hit=bool(job.cache_hit),
        )
        return _to_plain(df)

    def execute(self, sql: str) -> None:
        self.query(sql)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()

    def ensure_dataset(self) -> None:
        from google.cloud import bigquery

        ds = bigquery.Dataset(f"{self.project}.{self.dataset}")
        ds.location = self.location
        ds.description = "Olist marts loaded by olist-ecommerce-analytics (BigQuery sandbox)."
        self.client.create_dataset(ds, exists_ok=True)

    def load_parquet(self, table: str, path: Path) -> int:
        """Replace ``table`` with the rows of a local Parquet file (a free load job)."""
        from google.cloud import bigquery

        config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.PARQUET,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        )
        with open(path, "rb") as fh:
            job = self.client.load_table_from_file(
                fh, f"{self.project}.{self.dataset}.{table}", job_config=config
            )
        job.result()
        return int(self.client.get_table(f"{self.project}.{self.dataset}.{table}").num_rows)


def export_marts(duck: DuckDBEngine, out_dir: Path, tables: tuple[str, ...] = MART_TABLES) -> dict:
    """Write each mart to Parquet; returns {table: path}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for t in tables:
        path = out_dir / f"{t}.parquet"
        duck.con.execute(f"COPY (SELECT * FROM {t}) TO '{path.as_posix()}' (FORMAT PARQUET)")
        paths[t] = path
    return paths

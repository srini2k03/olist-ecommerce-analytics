"""Discover, run, snapshot and cross-check the queries in ``sql/``."""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from olist.config import SQL_DIR, SQL_RESULTS_DIR
from olist.db import Engine

REQUIRED_HEADERS = ("Business question:", "Method:", "Expected output:")


@dataclass(frozen=True)
class Query:
    path: Path

    @property
    def name(self) -> str:
        return self.path.stem

    @property
    def sql(self) -> str:
        return self.path.read_text(encoding="utf-8")

    @property
    def header(self) -> dict[str, str]:
        """Parse the leading ``-- Key: value`` comment block into a dict."""
        fields: dict[str, str] = {}
        key = None
        for line in self.sql.splitlines():
            if not line.startswith("--"):
                break
            text = line[2:].strip()
            m = re.match(r"^([A-Z][A-Za-z ]+):\s*(.*)$", text)
            if m:
                key = m.group(1)
                fields[key] = m.group(2)
            elif key:
                fields[key] += " " + text
        return fields


def list_queries(sql_dir: Path = SQL_DIR) -> list[Query]:
    return [Query(p) for p in sorted(Path(sql_dir).glob("[0-9][0-9]_*.sql"))]


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Make results comparable across engines (dates as ISO strings, numbers as float)."""
    out = df.copy()
    for c in out.columns:
        s = out[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            out[c] = s.dt.strftime("%Y-%m-%d")
        elif s.dtype == object and s.map(lambda v: isinstance(v, dt.date)).any():
            out[c] = s.map(lambda v: v.isoformat()[:10] if isinstance(v, dt.date) else v)
        elif pd.api.types.is_bool_dtype(s):
            out[c] = s.astype(int)
        elif pd.api.types.is_numeric_dtype(s):
            out[c] = s.astype(float)
        else:
            out[c] = s.astype(object).where(s.notna(), None)
    return out.reset_index(drop=True)


def run(engine: Engine, query: Query) -> pd.DataFrame:
    return normalise(engine.query(query.sql))


def frames_match(a: pd.DataFrame, b: pd.DataFrame, rtol: float = 1e-6) -> tuple[bool, str]:
    """Compare two normalised results cell by cell with a float tolerance."""
    if list(a.columns) != list(b.columns):
        return False, f"columns differ: {list(a.columns)} vs {list(b.columns)}"
    if a.shape != b.shape:
        return False, f"shape differs: {a.shape} vs {b.shape}"
    for c in a.columns:
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            ok = np.isclose(
                x.to_numpy(float), y.to_numpy(float), rtol=rtol, atol=1e-9, equal_nan=True
            )
            if not ok.all():
                i = int(np.argmin(ok))
                return False, f"column {c} row {i}: {x.iloc[i]!r} vs {y.iloc[i]!r}"
        else:
            xs, ys = x.astype(str).tolist(), y.astype(str).tolist()
            if xs != ys:
                i = next(k for k, (p, q) in enumerate(zip(xs, ys, strict=True)) if p != q)
                return False, f"column {c} row {i}: {xs[i]!r} vs {ys[i]!r}"
    return True, "ok"


def snapshot(df: pd.DataFrame, name: str, out_dir: Path = SQL_RESULTS_DIR) -> Path:
    """Write a rounded CSV snapshot so diffs in git stay readable."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.csv"
    out = df.round(6)
    for c in out.columns:
        col = out[c]
        vals = col.dropna()
        if pd.api.types.is_float_dtype(col) and len(vals) and (vals == vals.round()).all():
            out[c] = col.astype("Int64")
    out.to_csv(path, index=False)
    return path

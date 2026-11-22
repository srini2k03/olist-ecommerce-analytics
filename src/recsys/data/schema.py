"""Column dtypes of the processed H&M tables and parquet I/O.

The raw ``customer_id`` is a 64-character hex string. Keeping 31.8M of those in
memory costs several GB, so the processed tables store a 64-bit integer: the
last 16 hex digits read as an unsigned integer and reinterpreted as int64.
``customer_id_map.parquet`` keeps the original strings for a later submission.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TRANSACTION_DTYPES: dict[str, str] = {
    "t_dat": "datetime64[ns]",
    "customer_id": "int64",
    "article_id": "int32",
    "price": "float32",
    "sales_channel_id": "int8",
}

ARTICLE_DTYPES: dict[str, str] = {
    "article_id": "int32",
    "product_code": "int32",
    "prod_name": "str",
    "product_type_no": "int32",
    "product_type_name": "category",
    "product_group_name": "category",
    "graphical_appearance_no": "int32",
    "graphical_appearance_name": "category",
    "colour_group_code": "int32",
    "colour_group_name": "category",
    "perceived_colour_value_id": "int32",
    "perceived_colour_value_name": "category",
    "perceived_colour_master_id": "int32",
    "perceived_colour_master_name": "category",
    "department_no": "int32",
    "department_name": "category",
    "index_code": "category",
    "index_name": "category",
    "index_group_no": "int32",
    "index_group_name": "category",
    "section_no": "int32",
    "section_name": "category",
    "garment_group_no": "int32",
    "garment_group_name": "category",
    "detail_desc": "str",
}

CUSTOMER_DTYPES: dict[str, str] = {
    "customer_id": "int64",
    "FN": "float32",
    "Active": "float32",
    "club_member_status": "category",
    "fashion_news_frequency": "category",
    "age": "float32",
    "postal_code": "str",
}

TABLE_DTYPES = {
    "transactions": TRANSACTION_DTYPES,
    "articles": ARTICLE_DTYPES,
    "customers": CUSTOMER_DTYPES,
}


def hash_customer_ids(hex_ids: Iterable[str]) -> np.ndarray:
    """Map hex customer ids to int64 (last 16 hex digits, two's complement)."""
    values = [int(h[-16:], 16) for h in hex_ids]
    return np.array(values, dtype=np.uint64).view(np.int64)


def hash_customer_series(hex_ids: pd.Series) -> np.ndarray:
    """Vectorised over unique values: factorise first, hash each distinct id once."""
    codes, uniques = pd.factorize(hex_ids, sort=False)
    return hash_customer_ids(uniques)[codes]


def coerce(df: pd.DataFrame, table: str) -> pd.DataFrame:
    """Return ``df`` with the canonical column order and dtypes of ``table``."""
    dtypes = TABLE_DTYPES[table]
    missing = set(dtypes) - set(df.columns)
    if missing:
        raise ValueError(f"{table} is missing columns: {sorted(missing)}")
    out = df[list(dtypes)].copy()
    for col, dtype in dtypes.items():
        if str(out[col].dtype) != dtype:
            out[col] = out[col].astype(dtype)
    if table == "transactions":
        out["t_dat"] = out["t_dat"].dt.normalize()
    return out.reset_index(drop=True)


@dataclass
class HMData:
    """The three H&M tables in their processed form."""

    transactions: pd.DataFrame
    articles: pd.DataFrame
    customers: pd.DataFrame

    def coerced(self) -> HMData:
        return HMData(
            coerce(self.transactions, "transactions"),
            coerce(self.articles, "articles"),
            coerce(self.customers, "customers"),
        )

    def counts(self) -> dict[str, int]:
        return {
            "transactions": len(self.transactions),
            "articles": len(self.articles),
            "customers": len(self.customers),
            "active_customers": int(self.transactions["customer_id"].nunique()),
            "active_articles": int(self.transactions["article_id"].nunique()),
        }

    def write(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in TABLE_DTYPES:
            getattr(self, name).to_parquet(out_dir / f"{name}.parquet", index=False)

    @classmethod
    def read(cls, in_dir: Path) -> HMData:
        paths = {name: in_dir / f"{name}.parquet" for name in TABLE_DTYPES}
        absent = [str(p) for p in paths.values() if not p.exists()]
        if absent:
            raise FileNotFoundError(f"missing parquet files: {absent}")
        frames = {name: coerce(pd.read_parquet(p), name) for name, p in paths.items()}
        return cls(**frames)


def _clean(obj):
    """NaN/inf -> None and numpy scalars -> Python, so the output is strict JSON."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(payload), indent=2, default=str, allow_nan=False) + "\n")

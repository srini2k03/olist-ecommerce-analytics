"""Load the prebuilt bundle in app/data and apply the sidebar filters.

Filters: purchase-month range and customer state select orders; category, seller state and
seller select items. When an item filter is active, money metrics are summed over the
matching items and order metrics count the orders containing them (see olist.metrics).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import streamlit as st

DATA = Path(__file__).resolve().parents[1] / "data"
DEFAULT_START, DEFAULT_END = "2017-01", "2018-08"


@dataclass(frozen=True)
class Filters:
    start: str = DEFAULT_START  # YYYY-MM, inclusive
    end: str = DEFAULT_END
    customer_states: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    seller_states: tuple[str, ...] = ()
    sellers: tuple[int, ...] = field(default=())

    @property
    def item_filter(self) -> bool:
        return bool(self.categories or self.seller_states or self.sellers)

    def describe(self) -> str:
        parts = [f"{self.start} to {self.end}"]
        if self.customer_states:
            parts.append("customers in " + ", ".join(self.customer_states))
        if self.categories:
            parts.append(
                f"{len(self.categories)} categor{'y' if len(self.categories) == 1 else 'ies'}"
            )
        if self.seller_states:
            parts.append("sellers in " + ", ".join(self.seller_states))
        if self.sellers:
            parts.append(f"{len(self.sellers)} seller(s)")
        return " · ".join(parts)


def _json(name: str) -> dict:
    p = DATA / name
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_resource(show_spinner="Loading the data bundle…")
def load() -> dict:
    """Read every bundle file once per session (shared, never mutated)."""
    orders = pd.read_parquet(DATA / "orders.parquet")
    orders["gmv"] = orders["item_revenue"] + orders["freight_value"]
    items = pd.read_parquet(DATA / "items.parquet")
    cols = [
        "is_valid",
        "is_cancelled",
        "is_delivered",
        "is_late",
        "purchase_date",
        "purchase_month",
        "customer_state",
        "customer_unique_id",
        "review_score",
        "delivery_days",
    ]
    attrs = orders[cols].iloc[items["order_idx"].to_numpy()].reset_index(drop=True)
    items = pd.concat([items.reset_index(drop=True), attrs], axis=1)
    items["gmv"] = items["item_revenue"] + items["freight_value"]
    items["order_id"] = items["order_idx"]
    orders["order_id"] = orders.index.astype("int32")
    sellers = pd.read_parquet(DATA / "sellers.parquet")
    items["seller_state"] = items["seller_id"].map(sellers.set_index("code")["state"])
    months = pd.to_datetime(orders["purchase_month"]).dt.strftime("%Y-%m")
    return {
        "orders": orders,
        "items": items,
        "sellers": sellers,
        "months": sorted(months.unique()),
        "metrics_spec": _json("metrics.json"),
        "dq": _json("data_quality.json"),
        "findings": _json("findings.json"),
        "clv": _json("clv.json"),
        "delivery_model": _json("delivery_model.json"),
        "seller_ranking": _json("seller_ranking.json"),
        "ab_test": _json("ab_test.json"),
        "snapshot": _json("metrics_snapshot.json"),
        "model_card": _json("model_card.json"),
    }


def read_optional(name: str) -> pd.DataFrame | None:
    p = DATA / name
    return pd.read_parquet(p) if p.exists() else None


@st.cache_resource(show_spinner=False, max_entries=24)
def apply(f: Filters) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(orders, items) matching the filters. Cached per filter combination."""
    d = load()
    o, it = d["orders"], d["items"]
    lo = pd.Timestamp(f"{f.start}-01")
    hi = pd.Timestamp(f"{f.end}-01") + pd.offsets.MonthBegin(1)
    month = pd.to_datetime(o["purchase_month"])
    mask = (month >= lo) & (month < hi)
    if f.customer_states:
        mask &= o["customer_state"].isin(f.customer_states)
    o = o[mask]
    it = it[it["order_idx"].isin(o.index)]
    if f.categories:
        it = it[it["category"].isin(f.categories)]
    if f.seller_states:
        it = it[it["seller_state"].isin(f.seller_states)]
    if f.sellers:
        it = it[it["seller_id"].isin(f.sellers)]
    if f.item_filter:
        o = o[o.index.isin(it["order_idx"])]
    return o, it


def seller_label(d: dict, code: int) -> str:
    s = d["sellers"].set_index("code")
    if code not in s.index:
        return str(code)
    row = s.loc[code]
    return f"{row['seller_id'][:8]}… ({row['state']}, {row['city']})"

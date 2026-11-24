"""The metric layer in pandas.

``metrics/metrics.yml`` defines every metric once (label, unit, definition, caveats and the
SQL over the marts). This module implements the same definitions on an order-level frame
(``fct_orders`` columns) and, optionally, an item-level frame (``fct_order_items`` columns).
It depends only on pandas and numpy because the in-browser dashboard (Pyodide) imports this
exact module; ``tests/test_metrics.py`` requires the pandas and SQL values to agree.

Filters: when the caller filters items (by category or seller), money metrics are summed
over the matching items and order metrics count the orders that contain them. Payment value
cannot be split by item, so under an item filter it covers the whole matching orders.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

UNITS = ("BRL", "count", "share", "days", "stars")
KINDS = ("observed", "estimate")
DOMAINS = ("revenue", "order", "payment", "customer", "delivery", "review", "product", "seller")


@dataclass(frozen=True)
class MetricSpec:
    name: str
    label: str
    domain: str
    unit: str
    kind: str
    definition: str
    caveats: str = ""
    assumptions: tuple[str, ...] = ()
    sql: dict = field(default_factory=dict)


def load_spec(path: str | Path) -> dict:
    """Read the metric spec from YAML (pipeline) or JSON (dashboard bundle)."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        return json.loads(text)
    import yaml  # only the pipeline needs PyYAML

    return yaml.safe_load(text)


def metric_specs(spec: dict) -> dict[str, MetricSpec]:
    out = {}
    for m in spec["metrics"]:
        out[m["name"]] = MetricSpec(
            name=m["name"],
            label=m["label"],
            domain=m["domain"],
            unit=m["unit"],
            kind=m["kind"],
            definition=" ".join(m["definition"].split()),
            caveats=" ".join(m.get("caveats", "").split()),
            assumptions=tuple(m.get("assumptions", ())),
            sql=m.get("sql", {}),
        )
    return out


def assumption_values(spec: dict) -> dict[str, float]:
    return {k: float(v["value"]) for k, v in spec.get("assumptions", {}).items()}


def metric_sql(spec: MetricSpec, assumptions: dict[str, float]) -> str:
    """The SQL that computes one metric over the marts (one row, column ``value``)."""
    s = spec.sql
    if "query" in s:
        sql = s["query"]
    else:
        where = f" WHERE {s['where']}" if s.get("where") else ""
        sql = f"SELECT {s['expr']} AS value FROM {s['table']}{where}"
    return sql.format(**assumptions)


# ---------------------------------------------------------------------------- helpers
def _flag(s: pd.Series) -> pd.Series:
    return s.fillna(0).astype("int8").eq(1)


def _valid(o: pd.DataFrame) -> pd.DataFrame:
    return o[_flag(o["is_valid"])]


def _money_frame(o: pd.DataFrame, items: pd.DataFrame | None) -> pd.DataFrame:
    return _valid(items) if items is not None else _valid(o)


def _ratio(a: float, b: float) -> float:
    return float(a) / float(b) if b else math.nan


def _mean(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.mean()) if len(s) else math.nan


def _delivered(o: pd.DataFrame) -> pd.DataFrame:
    return o[_flag(o["is_valid"]) & _flag(o["is_delivered"])]


def _reviewed(o: pd.DataFrame) -> pd.DataFrame:
    v = _valid(o)
    return v[v["review_score"].notna()]


def monthly_gmv(o: pd.DataFrame, items: pd.DataFrame | None = None) -> pd.Series:
    f = _money_frame(o, items)
    return f.groupby("purchase_month", observed=True)["gmv"].sum().sort_index()


# ---------------------------------------------------------------------------- metric functions
Fn = Callable[[pd.DataFrame, "pd.DataFrame | None", dict], float]


def _sum(col: str) -> Fn:
    def f(o, i, a):
        frame = _money_frame(o, i) if col != "payment_value" else _valid(o)
        return float(frame[col].sum())

    return f


def _mom(o, i, a):
    m = monthly_gmv(o, i)
    return _ratio(m.iloc[-1], m.iloc[-2]) - 1 if len(m) >= 2 else math.nan


def _repeat_rate(o, i, a):
    v = _valid(o)
    seq = v["customer_order_seq"]
    firsts = v.loc[seq.eq(1), "customer_unique_id"]
    seconds = v.loc[seq.eq(2), "customer_unique_id"]
    return float(firsts.isin(seconds).mean()) if len(firsts) else math.nan


def _review_gap(o, i, a):
    d = _delivered(o)
    d = d[d["review_score"].notna()]
    late = _flag(d["is_late"])
    return _mean(d.loc[~late, "review_score"]) - _mean(d.loc[late, "review_score"])


REGISTRY: dict[str, Fn] = {
    "item_revenue": _sum("item_revenue"),
    "freight_value": _sum("freight_value"),
    "gmv": _sum("gmv"),
    "payment_value": _sum("payment_value"),
    "aov": lambda o, i, a: _ratio(_sum("gmv")(o, i, a), len(_valid(o))),
    "freight_share_of_gmv": lambda o, i, a: _ratio(
        _sum("freight_value")(o, i, a), _sum("gmv")(o, i, a)
    ),
    "gmv_mom_growth": _mom,
    "marketplace_revenue_estimate": lambda o, i, a: a["take_rate"] * _sum("item_revenue")(o, i, a),
    "contribution_estimate": lambda o, i, a: (
        a["take_rate"] * _sum("item_revenue")(o, i, a)
        - a["payment_cost_rate"] * _sum("payment_value")(o, i, a)
    ),
    "orders": lambda o, i, a: float(len(_valid(o))),
    "orders_placed": lambda o, i, a: float(len(o)),
    "cancellation_rate": lambda o, i, a: _ratio(_flag(o["is_cancelled"]).sum(), len(o)),
    "delivery_completion_rate": lambda o, i, a: _ratio(_flag(o["is_delivered"]).sum(), len(o)),
    "items_per_order": lambda o, i, a: _ratio(_valid(o)["n_items"].sum(), len(_valid(o))),
    "credit_card_share": lambda o, i, a: _mean(
        _valid(o)["main_payment_type"].dropna().eq("credit_card").astype(float)
    ),
    "avg_installments": lambda o, i, a: _mean(_valid(o)["installments"]),
    "customers": lambda o, i, a: float(_valid(o)["customer_unique_id"].nunique()),
    "new_customers": lambda o, i, a: float(_valid(o)["customer_order_seq"].eq(1).sum()),
    "returning_customers": lambda o, i, a: float(
        _valid(o).loc[_valid(o)["customer_order_seq"].gt(1), "customer_unique_id"].nunique()
    ),
    "repeat_rate": _repeat_rate,
    "gmv_per_customer": lambda o, i, a: _ratio(
        _sum("gmv")(o, i, a), _valid(o)["customer_unique_id"].nunique()
    ),
    "orders_per_customer": lambda o, i, a: _ratio(
        len(_valid(o)), _valid(o)["customer_unique_id"].nunique()
    ),
    "avg_delivery_days": lambda o, i, a: _mean(_delivered(o)["delivery_days"]),
    "avg_promised_days": lambda o, i, a: _mean(_delivered(o)["estimated_days"]),
    "late_rate": lambda o, i, a: _mean(_delivered(o)["is_late"]),
    "avg_days_late_when_late": lambda o, i, a: _mean(
        _valid(o).loc[_flag(_valid(o)["is_late"]), "delay_days"]
    ),
    "avg_review": lambda o, i, a: _mean(_reviewed(o)["review_score"]),
    "bad_review_rate": lambda o, i, a: _mean(_reviewed(o)["review_score"].le(2).astype(float)),
    "review_gap_late": _review_gap,
    "active_categories": lambda o, i, a: float(_valid(_items_or_fail(i))["category"].nunique()),
    "active_sellers": lambda o, i, a: float(_valid(_items_or_fail(i))["seller_id"].nunique()),
    "item_revenue_per_seller": lambda o, i, a: _ratio(
        _valid(_items_or_fail(i))["item_revenue"].sum(),
        _valid(_items_or_fail(i))["seller_id"].nunique(),
    ),
}


def _items_or_fail(items: pd.DataFrame | None) -> pd.DataFrame:
    if items is None:
        raise ValueError("this metric needs the item-level frame")
    return items


def compute(
    name: str,
    orders: pd.DataFrame,
    items: pd.DataFrame | None = None,
    assumptions: dict[str, float] | None = None,
) -> float:
    """One metric on (already filtered) frames."""
    if name not in REGISTRY:
        raise KeyError(f"unknown metric {name!r}")
    return REGISTRY[name](orders, items, assumptions or {})


def compute_all(
    orders: pd.DataFrame,
    items: pd.DataFrame | None = None,
    assumptions: dict[str, float] | None = None,
    names: list[str] | None = None,
) -> dict[str, float]:
    return {n: compute(n, orders, items, assumptions) for n in (names or list(REGISTRY))}


# ---------------------------------------------------------------------------- formatting
def format_value(value: float, unit: str, compact: bool = True) -> str:
    """Human-readable value: R$ 15.7M, 6.8%, 12.1 days, 4.09★, 97,905."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    if unit == "BRL":
        if compact and abs(value) >= 1e6:
            return f"R$ {value / 1e6:,.2f}M"
        if compact and abs(value) >= 1e4:
            return f"R$ {value / 1e3:,.1f}k"
        return f"R$ {value:,.2f}"
    if unit == "share":
        return f"{value:.1%}" if abs(value) >= 0.001 else f"{value:.2%}"
    if unit == "days":
        return f"{value:.1f} days"
    if unit == "stars":
        return f"{value:.2f}★"
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.2f}"

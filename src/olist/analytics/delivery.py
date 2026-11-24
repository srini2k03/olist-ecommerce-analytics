"""Delivery performance breakdowns (browser-safe: pandas + numpy).

All breakdowns use delivered valid orders. ``late`` means delivered after the date promised
at checkout; days are whole calendar days.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from olist.stats.core import wilson_interval

FREIGHT_BANDS = [0, 10, 15, 20, 30, 50, np.inf]
DISTANCE_BANDS = [0, 100, 300, 600, 1000, 2000, np.inf]


def delivered(orders: pd.DataFrame) -> pd.DataFrame:
    return orders[orders["is_valid"].eq(1) & orders["is_delivered"].eq(1)]


def band(values: pd.Series, edges, unit: str = "") -> pd.Series:
    labels = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        labels.append(f"{lo:g}+{unit}" if not np.isfinite(hi) else f"{lo:g}-{hi:g}{unit}")
    return pd.cut(values, edges, labels=labels, right=False)


def breakdown(orders: pd.DataFrame, by: str | list[str], min_orders: int = 1) -> pd.DataFrame:
    """Per group: delivered orders, mean/p90 delivery days, mean promised days, late rate
    with a Wilson 95% interval, mean days late when late and mean review."""
    d = delivered(orders).copy()
    if d.empty:
        return pd.DataFrame()
    d["late_delay"] = d["delay_days"].where(d["is_late"].eq(1))
    g = d.groupby(by, observed=True)
    out = pd.DataFrame(
        {
            "orders": g.size(),
            "late_orders": g["is_late"].sum(),
            "avg_delivery_days": g["delivery_days"].mean(),
            "p90_delivery_days": g["delivery_days"].quantile(0.9),
            "avg_promised_days": g["estimated_days"].mean(),
            "avg_days_late_when_late": g["late_delay"].mean(),
            "avg_review": g["review_score"].mean(),
        }
    )
    out = out[out["orders"] >= min_orders]
    out["late_rate"] = out["late_orders"] / out["orders"]
    lo, hi = wilson_interval(out["late_orders"], out["orders"])
    out["late_rate_ci_low"], out["late_rate_ci_high"] = lo, hi
    out["avg_buffer_days"] = out["avg_promised_days"] - out["avg_delivery_days"]
    return out.reset_index()

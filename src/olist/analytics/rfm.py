"""Rule-based RFM scoring and segments (transparent, filter-friendly, browser-safe).

Recency and monetary are scored in quintiles (NTILE(5), ties broken by customer id exactly
as in SQL 03). Frequency is NOT a quintile: about 97% of customers ordered once, so it is
capped (1, 2, ... 5). Segment rules are listed in SEGMENT_RULES and mirrored in
sql/analytics/customer/03_rfm_scores.sql; a test requires both to agree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEGMENT_ORDER = [
    "Champions",
    "Loyal, lapsing",
    "New big spenders",
    "New low spenders",
    "Fading one-timers",
    "At-risk big spenders",
    "Lost one-timers",
]
SEGMENT_RULES = {
    "Champions": "2+ orders and recency score 4-5 (ordered recently)",
    "Loyal, lapsing": "2+ orders, recency score 1-3",
    "New big spenders": "1 order, recency 4-5, monetary 4-5",
    "New low spenders": "1 order, recency 4-5, monetary 1-3",
    "Fading one-timers": "1 order, recency 3",
    "At-risk big spenders": "1 order, recency 1-2, monetary 4-5",
    "Lost one-timers": "1 order, recency 1-2, monetary 1-3",
}
SEGMENT_ACTIONS = {
    "Champions": "Protect: early access, loyalty perks, referral asks. No discounts needed.",
    "Loyal, lapsing": "Win back with a personal reminder of the categories they buy.",
    "New big spenders": "Second-order nudge within 30-60 days; complementary category.",
    "New low spenders": "Cheap reactivation: free-shipping-threshold reminder, cross-sell.",
    "Fading one-timers": "Low-cost email touch before they lapse.",
    "At-risk big spenders": "Targeted win-back on the category they bought, before day 365.",
    "Lost one-timers": "Low-cost channels only; do not pay to reacquire.",
}


def ntile(n: int, k: int) -> np.ndarray:
    """SQL NTILE(k) bucket for positions 0..n-1 (first n % k buckets get one extra row)."""
    base, extra = divmod(n, k)
    sizes = [base + (1 if i < extra else 0) for i in range(k)]
    return np.repeat(np.arange(1, k + 1), sizes)


def rfm_table(orders: pd.DataFrame, as_of) -> pd.DataFrame:
    """Per customer, from valid orders purchased before ``as_of``: recency_days (days since
    the last order), frequency (orders), monetary (GMV), avg_review."""
    as_of = pd.Timestamp(as_of)
    o = orders[(orders["is_valid"] == 1) & (pd.to_datetime(orders["purchase_date"]) < as_of)]
    g = o.groupby("customer_unique_id", observed=True)
    last = pd.to_datetime(g["purchase_date"].max())
    out = pd.DataFrame(
        {
            "recency_days": (as_of - last).dt.days,
            "frequency": g.size(),
            "monetary": g["gmv"].sum(),
            "avg_review": g["review_score"].mean(),
        }
    )
    return out.reset_index()


def score(rfm: pd.DataFrame) -> pd.DataFrame:
    out = rfm.copy()
    n = len(out)
    by_r = out.sort_values(["recency_days", "customer_unique_id"], ascending=[False, True])
    out.loc[by_r.index, "r_score"] = ntile(n, 5)
    by_m = out.sort_values(["monetary", "customer_unique_id"], ascending=[True, True])
    out.loc[by_m.index, "m_score"] = ntile(n, 5)
    out["f_score"] = out["frequency"].clip(upper=5)
    for c in ("r_score", "m_score", "f_score"):
        out[c] = out[c].astype(int)
    f, r, m = out.f_score, out.r_score, out.m_score
    conditions = [
        (f >= 2) & (r >= 4),
        f >= 2,
        (r >= 4) & (m >= 4),
        r >= 4,
        (r <= 2) & (m >= 4),
        r <= 2,
    ]
    names = [
        "Champions",
        "Loyal, lapsing",
        "New big spenders",
        "New low spenders",
        "At-risk big spenders",
        "Lost one-timers",
    ]
    out["segment"] = np.select(conditions, names, default="Fading one-timers")
    return out


def segment_summary(scored: pd.DataFrame) -> pd.DataFrame:
    """Size, revenue share, value and repeat behaviour per segment, in SEGMENT_ORDER."""
    if scored.empty:
        return pd.DataFrame()
    g = scored.groupby("segment")
    s = pd.DataFrame(
        {
            "customers": g.size(),
            "revenue": g["monetary"].sum(),
            "orders": g["frequency"].sum(),
            "avg_recency_days": g["recency_days"].mean(),
            "avg_frequency": g["frequency"].mean(),
            "avg_monetary": g["monetary"].mean(),
            "repeat_rate": g["frequency"].apply(lambda x: float((x >= 2).mean())),
            "avg_review": g["avg_review"].mean(),
        }
    )
    s["customer_share"] = s["customers"] / s["customers"].sum()
    s["revenue_share"] = s["revenue"] / s["revenue"].sum()
    s["aov"] = s["revenue"] / s["orders"]
    s["rule"] = s.index.map(SEGMENT_RULES)
    s["action"] = s.index.map(SEGMENT_ACTIONS)
    order = [x for x in SEGMENT_ORDER if x in s.index]
    return s.loc[order].reset_index()

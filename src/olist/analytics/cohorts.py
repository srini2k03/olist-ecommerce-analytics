"""Acquisition cohorts: retention matrix, cohort revenue and retention curves.

A customer's cohort is the month of their first valid order ever (``cohort_month`` in
fct_orders). Retention at month k = share of the cohort with a valid order k months later.
Cohorts are only averaged at ages they have actually been observed for, so young cohorts do
not drag the curve down (right-censoring).
"""

from __future__ import annotations

import pandas as pd


def month_index(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.year * 12 + s.dt.month


def retention_matrix(orders: pd.DataFrame, max_months: int | None = None) -> pd.DataFrame:
    """Long table: cohort_month, months_since_first, active_customers, cohort_size,
    retention, gmv, cumulative_gmv_per_customer. ``orders`` = valid orders with
    customer_unique_id, purchase_month, cohort_month and gmv."""
    o = orders[["customer_unique_id", "purchase_month", "cohort_month", "gmv"]].dropna(
        subset=["cohort_month"]
    )
    if o.empty:
        return pd.DataFrame(
            columns=[
                "cohort_month",
                "months_since_first",
                "active_customers",
                "cohort_size",
                "retention",
                "gmv",
                "cumulative_gmv_per_customer",
            ]
        )
    k = month_index(o["purchase_month"]) - month_index(o["cohort_month"])
    o = o.assign(months_since_first=k.astype(int), cohort_month=pd.to_datetime(o.cohort_month))
    if max_months is not None:
        o = o[o.months_since_first <= max_months]
    g = o.groupby(["cohort_month", "months_since_first"], observed=True)
    m = g.agg(active_customers=("customer_unique_id", "nunique"), gmv=("gmv", "sum"))
    m = m.reset_index()
    size = m[m.months_since_first == 0].set_index("cohort_month")["active_customers"]
    m["cohort_size"] = m["cohort_month"].map(size)
    m = m[m["cohort_size"].notna()].copy()
    m["cohort_size"] = m["cohort_size"].astype(int)
    m["retention"] = m["active_customers"] / m["cohort_size"]
    m = m.sort_values(["cohort_month", "months_since_first"])
    m["cumulative_gmv_per_customer"] = m.groupby("cohort_month")["gmv"].cumsum() / m.cohort_size
    return m.reset_index(drop=True)


def retention_curve(matrix: pd.DataFrame, last_month=None, max_months: int = 12) -> pd.DataFrame:
    """Size-weighted retention by months since first order, using only cohorts observed for
    at least that many months (``last_month`` defaults to the latest month in the matrix)."""
    if matrix.empty:
        return pd.DataFrame(columns=["months_since_first", "retention", "cohorts", "customers"])
    cohorts = matrix.drop_duplicates("cohort_month")[["cohort_month", "cohort_size"]]
    cohort_idx = month_index(cohorts.cohort_month)
    if last_month is None:
        last_idx = int((month_index(matrix.cohort_month) + matrix.months_since_first).max())
    else:
        last_idx = int(month_index(pd.Series([pd.Timestamp(last_month)])).iloc[0])
    cohorts = cohorts.assign(age=last_idx - cohort_idx)
    active = matrix.pivot_table(
        index="cohort_month",
        columns="months_since_first",
        values="active_customers",
        aggfunc="sum",
        fill_value=0,
    )
    rows = []
    for k in range(0, max_months + 1):
        eligible = cohorts[cohorts.age >= k]
        if eligible.empty:
            break
        act = active.reindex(eligible.cohort_month)[k].sum() if k in active.columns else 0
        size = eligible.cohort_size.sum()
        rows.append(
            {
                "months_since_first": k,
                "retention": act / size,
                "cohorts": len(eligible),
                "customers": int(size),
            }
        )
    return pd.DataFrame(rows)


def cohort_summary(matrix: pd.DataFrame) -> pd.DataFrame:
    """Per cohort: size, month-0 GMV, later GMV, GMV per customer, share of GMV after month 0,
    and the share of the cohort that ever came back (any month > 0)."""
    if matrix.empty:
        return pd.DataFrame()
    g = matrix.groupby("cohort_month")
    out = pd.DataFrame(
        {
            "cohort_size": g["cohort_size"].first(),
            "gmv_month_0": matrix[matrix.months_since_first == 0].set_index("cohort_month")["gmv"],
            "gmv_total": g["gmv"].sum(),
        }
    )
    out["gmv_later_months"] = out["gmv_total"] - out["gmv_month_0"]
    out["gmv_per_customer"] = out["gmv_total"] / out["cohort_size"]
    out["share_gmv_after_month_0"] = out["gmv_later_months"] / out["gmv_total"]
    return out.reset_index()

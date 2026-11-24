"""Analysis-ready frames pulled from the DuckDB warehouse.

They read the marts built by ``python -m olist.pipeline`` and are shared by the notebooks,
the churn model and the analytics/app builds.
"""

from __future__ import annotations

import duckdb
import pandas as pd

from olist.config import WAREHOUSE_PATH

ORDER_FACTS_SQL = """
-- Valid orders (not cancelled/unavailable, with items) from the fct_orders mart, with the
-- column names the notebooks and the churn model use. order_value is GMV (items + freight).
SELECT
    order_id,
    customer_unique_id,
    customer_state,
    order_status,
    purchased_at,
    delivered_at,
    estimated_delivery_at AS estimated_at,
    n_items,
    item_revenue AS items_value,
    freight_value,
    gmv AS order_value,
    payment_value,
    installments,
    main_payment_type,
    used_voucher,
    category,
    seller_state,
    delivery_days,
    delay_days,
    review_score,
    review_sent_at AS review_created_at,
    review_answered_at,
    has_comment
FROM fct_orders
WHERE is_valid = 1
"""


def connect(path=WAREHOUSE_PATH) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(path), read_only=True)


def order_facts(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """One row per valid order with value, delivery, review and customer attributes."""
    df = con.execute(ORDER_FACTS_SQL).df()
    df["is_late"] = (df["delay_days"] > 0).astype("float").where(df["delay_days"].notna())
    return df.sort_values(["purchased_at", "order_id"]).reset_index(drop=True)


def run_sql_file(con: duckdb.DuckDBPyConnection, name: str) -> pd.DataFrame:
    """Run one query from ``sql/`` by file stem (e.g. ``"01_monthly_revenue_mom"``)."""
    from olist.sql_runner import list_queries

    q = {q.name: q for q in list_queries()}[name]
    return con.execute(q.sql).df()

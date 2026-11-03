"""Analysis-ready frames pulled from the DuckDB warehouse.

These helpers use DuckDB SQL directly (they feed Python, not the portable
``sql/`` library) and are shared by notebooks, the model and the app build.
"""

from __future__ import annotations

import duckdb
import pandas as pd

from olist.config import WAREHOUSE_PATH

ORDER_FACTS_SQL = """
WITH items AS (
    SELECT order_id,
           COUNT(*) AS n_items,
           SUM(price) AS items_value,
           SUM(freight_value) AS freight_value,
           MIN(CASE WHEN order_item_id = 1 THEN product_id END) AS first_product_id,
           MIN(CASE WHEN order_item_id = 1 THEN seller_id END) AS first_seller_id
    FROM order_items GROUP BY order_id
),
pay AS (
    SELECT order_id,
           SUM(payment_value) AS payment_value,
           MAX(payment_installments) AS installments,
           arg_max(payment_type, payment_value) AS main_payment_type,
           MAX(CASE WHEN payment_type = 'voucher' THEN 1 ELSE 0 END) AS used_voucher
    FROM order_payments GROUP BY order_id
),
rev AS (
    SELECT order_id,
           AVG(review_score) AS review_score,
           MIN(review_creation_date) AS review_created_at,
           MAX(CASE WHEN review_comment_message IS NOT NULL THEN 1 ELSE 0 END) AS has_comment
    FROM order_reviews GROUP BY order_id
)
SELECT
    o.order_id,
    c.customer_unique_id,
    c.customer_state,
    o.order_status,
    o.order_purchase_timestamp AS purchased_at,
    o.order_delivered_customer_date AS delivered_at,
    o.order_estimated_delivery_date AS estimated_at,
    i.n_items,
    i.items_value,
    i.freight_value,
    i.items_value + i.freight_value AS order_value,
    p.payment_value,
    p.installments,
    p.main_payment_type,
    p.used_voucher,
    COALESCE(t.product_category_name_english, pr.product_category_name, 'unknown') AS category,
    s.seller_state,
    date_diff('day', CAST(o.order_purchase_timestamp AS DATE),
              CAST(o.order_delivered_customer_date AS DATE)) AS delivery_days,
    date_diff('day', CAST(o.order_estimated_delivery_date AS DATE),
              CAST(o.order_delivered_customer_date AS DATE)) AS delay_days,
    r.review_score,
    r.review_created_at,
    r.has_comment
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
LEFT JOIN items i ON i.order_id = o.order_id
LEFT JOIN pay p ON p.order_id = o.order_id
LEFT JOIN rev r ON r.order_id = o.order_id
LEFT JOIN products pr ON pr.product_id = i.first_product_id
LEFT JOIN category_translation t ON t.product_category_name = pr.product_category_name
LEFT JOIN sellers s ON s.seller_id = i.first_seller_id
WHERE o.order_status NOT IN ('canceled', 'unavailable')
  AND i.n_items IS NOT NULL
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

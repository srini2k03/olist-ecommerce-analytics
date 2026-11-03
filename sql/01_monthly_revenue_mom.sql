-- 01_monthly_revenue_mom.sql
-- Business question: How fast is the marketplace growing month over month, and is
--   growth coming from more orders or from bigger orders?
-- Method: Keep orders that were not canceled/unavailable, purchased Jan 2017 to Aug 2018
--   (the edge months are partial). Order value = item price + freight from order_items.
--   Aggregate by purchase month, then use LAG for MoM and YoY growth and a 3-month
--   moving average window.
-- Expected output: one row per month (20 rows): month, orders, gmv, aov, mom_growth,
--   yoy_growth (NULL for the first 12 months), gmv_3m_avg.
-- Dialect notes: CAST(date_trunc(...) AS DATE) gives the same type in DuckDB and Postgres.
WITH valid_orders AS (
    SELECT
        o.order_id,
        CAST(date_trunc('month', o.order_purchase_timestamp) AS DATE) AS month
    FROM orders AS o
    WHERE o.order_status NOT IN ('canceled', 'unavailable')
      AND o.order_purchase_timestamp >= DATE '2017-01-01'
      AND o.order_purchase_timestamp < DATE '2018-09-01'
),
order_value AS (
    SELECT
        oi.order_id,
        SUM(oi.price) AS items_value,
        SUM(oi.freight_value) AS freight_value
    FROM order_items AS oi
    GROUP BY oi.order_id
),
monthly AS (
    SELECT
        v.month,
        COUNT(*) AS orders,
        SUM(ov.items_value + ov.freight_value) AS gmv
    FROM valid_orders AS v
    JOIN order_value AS ov ON ov.order_id = v.order_id
    GROUP BY v.month
)
SELECT
    month,
    orders,
    gmv,
    gmv / orders AS aov,
    gmv / LAG(gmv) OVER (ORDER BY month) - 1 AS mom_growth,
    gmv / LAG(gmv, 12) OVER (ORDER BY month) - 1 AS yoy_growth,
    AVG(gmv) OVER (ORDER BY month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS gmv_3m_avg
FROM monthly
ORDER BY month;

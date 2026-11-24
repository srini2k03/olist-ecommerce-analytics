-- 01_monthly_revenue_mom.sql
-- Business question: How fast is the marketplace growing month over month, and is
--   growth coming from more orders or from bigger orders?
-- Method: Valid orders (fct_orders.is_valid: not cancelled/unavailable, has items)
--   purchased Jan 2017 to Aug 2018 (the edge months are partial), aggregated by purchase
--   month. Item revenue, freight and GMV (= item revenue + freight) are reported separately.
--   LAG gives MoM and YoY growth of GMV and a window gives the 3-month moving average.
-- Expected output: one row per month (20 rows): month, orders, item_revenue, freight_value,
--   gmv, aov (GMV per order), mom_growth, yoy_growth (NULL for the first 12 months),
--   gmv_3m_avg.
-- Dialect notes: purchase_month is a DATE built once in stg_orders.
WITH monthly AS (
    SELECT
        purchase_month AS month,
        COUNT(*) AS orders,
        SUM(item_revenue) AS item_revenue,
        SUM(freight_value) AS freight_value,
        SUM(gmv) AS gmv
    FROM fct_orders
    WHERE is_valid = 1
      AND purchase_date >= DATE '2017-01-01'
      AND purchase_date < DATE '2018-09-01'
    GROUP BY purchase_month
)
SELECT
    month,
    orders,
    item_revenue,
    freight_value,
    gmv,
    gmv / orders AS aov,
    gmv / LAG(gmv) OVER (ORDER BY month) - 1 AS mom_growth,
    gmv / LAG(gmv, 12) OVER (ORDER BY month) - 1 AS yoy_growth,
    AVG(gmv) OVER (ORDER BY month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS gmv_3m_avg
FROM monthly
ORDER BY month;

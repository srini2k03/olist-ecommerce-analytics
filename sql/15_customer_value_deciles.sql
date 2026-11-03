-- 15_customer_value_deciles.sql
-- Business question: How concentrated is customer value? How much revenue comes from
--   the top 10% of customers?
-- Method: Lifetime spend (price + freight of valid orders) per customer_unique_id.
--   NTILE(10) splits customers into deciles (10 = highest spend); a running SUM over
--   deciles from the top down gives cumulative revenue share.
-- Expected output: 10 rows with customers, revenue, revenue_share, cumulative share from
--   the top, mean spend and repeat-customer share per decile.
-- Dialect notes: standard NTILE and window SUM.
WITH order_value AS (
    SELECT order_id, SUM(price + freight_value) AS order_value
    FROM order_items
    GROUP BY order_id
),
customer_value AS (
    SELECT
        c.customer_unique_id,
        COUNT(DISTINCT o.order_id) AS orders,
        SUM(ov.order_value) AS lifetime_value
    FROM orders AS o
    JOIN customers AS c ON c.customer_id = o.customer_id
    JOIN order_value AS ov ON ov.order_id = o.order_id
    WHERE o.order_status NOT IN ('canceled', 'unavailable')
    GROUP BY c.customer_unique_id
),
deciles AS (
    SELECT
        *,
        NTILE(10) OVER (ORDER BY lifetime_value, customer_unique_id) AS value_decile
    FROM customer_value
),
by_decile AS (
    SELECT
        value_decile,
        COUNT(*) AS customers,
        SUM(lifetime_value) AS revenue,
        AVG(lifetime_value) AS avg_lifetime_value,
        MIN(lifetime_value) AS min_lifetime_value,
        AVG(CASE WHEN orders > 1 THEN 1.0 ELSE 0.0 END) AS repeat_share
    FROM deciles
    GROUP BY value_decile
)
SELECT
    value_decile,
    customers,
    revenue,
    revenue / SUM(revenue) OVER () AS revenue_share,
    SUM(revenue) OVER (ORDER BY value_decile DESC ROWS UNBOUNDED PRECEDING)
        / SUM(revenue) OVER () AS cumulative_share_from_top,
    avg_lifetime_value,
    min_lifetime_value,
    repeat_share
FROM by_decile
ORDER BY value_decile DESC;

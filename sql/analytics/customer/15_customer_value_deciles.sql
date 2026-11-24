-- 15_customer_value_deciles.sql
-- Business question: How concentrated is customer value? What share of GMV comes from
--   the top 10% of customers?
-- Method: Lifetime GMV and order count per person from dim_customers (valid orders over
--   the full history), NTILE(10) deciles, then per-decile totals and the cumulative share
--   from the top decile down.
-- Expected output: 10 rows from decile 10 (highest value) to 1, with revenue_share
--   summing to 1 and the repeat share per decile.
-- Dialect notes: NTILE ordered by value and id for deterministic ties.
WITH customer_value AS (
    SELECT customer_unique_id, n_orders AS orders, lifetime_gmv AS lifetime_value
    FROM dim_customers
    WHERE n_orders > 0
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

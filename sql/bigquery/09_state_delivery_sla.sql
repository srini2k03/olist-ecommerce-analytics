-- 09_state_delivery_sla.sql (BigQuery version)
-- Same question, method and output as sql/analytics/delivery/09_state_delivery_sla.sql.
-- Dialect notes: BigQuery has no PERCENTILE_CONT ... WITHIN GROUP aggregate; the exact
--   percentile is the analytic PERCENTILE_CONT(x, p) OVER (PARTITION BY group), computed on
--   the filtered rows and then carried through the GROUP BY with ANY_VALUE (constant per
--   group). NULLs are ignored on all engines, and interpolation is linear on all engines.
WITH base AS (
    SELECT
        customer_state,
        delivery_days,
        estimated_days,
        is_late,
        review_score,
        PERCENTILE_CONT(delivery_days, 0.9) OVER (PARTITION BY customer_state) AS p90
    FROM fct_orders
    WHERE is_delivered = 1
),
by_state AS (
    SELECT
        customer_state,
        COUNT(*) AS orders,
        AVG(1.0 * delivery_days) AS avg_delivery_days,
        ANY_VALUE(p90) AS p90_delivery_days,
        AVG(1.0 * estimated_days) AS avg_promised_days,
        AVG(1.0 * is_late) AS late_rate,
        AVG(review_score) AS avg_review
    FROM base
    GROUP BY customer_state
    HAVING COUNT(*) >= 300
)
SELECT
    RANK() OVER (ORDER BY late_rate DESC) AS late_rank,
    customer_state,
    orders,
    avg_delivery_days,
    p90_delivery_days,
    avg_promised_days,
    avg_promised_days - avg_delivery_days AS avg_buffer_days,
    late_rate,
    avg_review
FROM by_state
ORDER BY late_rank, customer_state;

-- 12_time_to_second_order.sql (BigQuery version)
-- Same question, method and output as sql/analytics/customer/12_time_to_second_order.sql.
-- Dialect notes: DATE - DATE becomes DATE_DIFF(a, b, DAY); the overall median uses the
--   analytic PERCENTILE_CONT(x, 0.5) OVER () collapsed with ANY_VALUE.
WITH customer_orders AS (
    SELECT
        customer_unique_id,
        customer_order_seq,
        purchased_at,
        LEAD(purchased_at) OVER (
            PARTITION BY customer_unique_id ORDER BY purchased_at, order_id
        ) AS next_ts
    FROM fct_orders
    WHERE is_valid = 1
),
gaps AS (
    SELECT DATE_DIFF(CAST(next_ts AS DATE), CAST(purchased_at AS DATE), DAY) AS gap_days
    FROM customer_orders
    WHERE customer_order_seq = 1 AND next_ts IS NOT NULL
),
bucketed AS (
    SELECT
        CASE
            WHEN gap_days = 0 THEN 1
            WHEN gap_days <= 7 THEN 2
            WHEN gap_days <= 30 THEN 3
            WHEN gap_days <= 90 THEN 4
            WHEN gap_days <= 180 THEN 5
            WHEN gap_days <= 365 THEN 6
            ELSE 7
        END AS bucket_order,
        gap_days
    FROM gaps
),
median_gap AS (
    SELECT ANY_VALUE(m) AS median_gap_days
    FROM (SELECT PERCENTILE_CONT(gap_days, 0.5) OVER () AS m FROM gaps)
)
SELECT
    b.bucket_order,
    CASE b.bucket_order
        WHEN 1 THEN 'same day'
        WHEN 2 THEN '1-7 days'
        WHEN 3 THEN '8-30 days'
        WHEN 4 THEN '31-90 days'
        WHEN 5 THEN '91-180 days'
        WHEN 6 THEN '181-365 days'
        ELSE '365+ days'
    END AS gap_bucket,
    COUNT(*) AS customers,
    1.0 * COUNT(*) / SUM(COUNT(*)) OVER () AS share,
    1.0 * SUM(COUNT(*)) OVER (ORDER BY b.bucket_order ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
        / SUM(COUNT(*)) OVER () AS cumulative_share,
    MAX(m.median_gap_days) AS median_gap_days
FROM bucketed AS b
CROSS JOIN median_gap AS m
GROUP BY b.bucket_order
ORDER BY b.bucket_order;

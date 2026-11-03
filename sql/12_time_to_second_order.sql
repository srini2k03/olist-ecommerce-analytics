-- 12_time_to_second_order.sql
-- Business question: When customers do come back, how long does it take? This sets the
--   timing of win-back campaigns and the label window of the churn model.
-- Method: LEAD() over each customer's orders gives the next purchase time; for the first
--   order this is the time to the second order. Gaps are bucketed. Same-day second
--   orders are split out because many are one shopping session split into two orders,
--   not real loyalty.
-- Expected output: 7 buckets with counts, share and cumulative share, plus the median
--   gap (in days) of all second orders on every row.
-- Dialect notes: DATE - DATE; PERCENTILE_CONT as a plain aggregate in a CTE.
WITH customer_orders AS (
    SELECT
        c.customer_unique_id,
        o.order_purchase_timestamp,
        ROW_NUMBER() OVER (
            PARTITION BY c.customer_unique_id
            ORDER BY o.order_purchase_timestamp, o.order_id
        ) AS order_seq,
        LEAD(o.order_purchase_timestamp) OVER (
            PARTITION BY c.customer_unique_id
            ORDER BY o.order_purchase_timestamp, o.order_id
        ) AS next_ts
    FROM orders AS o
    JOIN customers AS c ON c.customer_id = o.customer_id
    WHERE o.order_status NOT IN ('canceled', 'unavailable')
),
gaps AS (
    SELECT CAST(next_ts AS DATE) - CAST(order_purchase_timestamp AS DATE) AS gap_days
    FROM customer_orders
    WHERE order_seq = 1 AND next_ts IS NOT NULL
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
    SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY gap_days) AS median_gap_days
    FROM gaps
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
    1.0 * SUM(COUNT(*)) OVER (ORDER BY b.bucket_order ROWS UNBOUNDED PRECEDING)
        / SUM(COUNT(*)) OVER () AS cumulative_share,
    MAX(m.median_gap_days) AS median_gap_days
FROM bucketed AS b
CROSS JOIN median_gap AS m
GROUP BY b.bucket_order
ORDER BY b.bucket_order;

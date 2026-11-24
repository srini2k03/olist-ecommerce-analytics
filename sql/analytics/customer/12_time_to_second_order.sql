-- 12_time_to_second_order.sql
-- Business question: When customers do come back, how long do they take?
-- Method: Valid orders over the full history. LEAD over each customer's orders (ordered
--   by purchase time) gives the next order; the gap from the first to the second order is
--   bucketed. Also reports the median gap with PERCENTILE_CONT.
-- Expected output: 7 bucket rows with customers, share, cumulative_share and the overall
--   median gap. A large same-day share reveals split baskets, not loyalty.
-- Dialect notes: DATE - DATE for whole days; PERCENTILE_CONT ... WITHIN GROUP on both.
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
    SELECT CAST(next_ts AS DATE) - CAST(purchased_at AS DATE) AS gap_days
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

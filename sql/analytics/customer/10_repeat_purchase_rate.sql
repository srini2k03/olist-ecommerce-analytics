-- 10_repeat_purchase_rate.sql
-- Business question: What share of new customers ever place a second order, and has that
--   improved over time?
-- Method: Customers acquired Jan 2017 to Aug 2018 (first valid order ever in that window).
--   A customer repeats if their second valid order (customer_order_seq = 2) is before
--   2018-09-01; repeat_rate_90d needs it within 90 days of the first. Grouped by the
--   quarter of the first order, plus an 'All' row.
-- Expected output: 7 quarterly rows + 'All'. Overall repeat rate about 3%.
-- Dialect notes: quarter label built with EXTRACT + string concatenation (||), both
--   engines.
WITH firsts AS (
    SELECT customer_unique_id, purchased_at AS first_ts
    FROM fct_orders
    WHERE is_valid = 1
      AND customer_order_seq = 1
      AND purchase_date >= DATE '2017-01-01'
      AND purchase_date < DATE '2018-09-01'
),
seconds AS (
    SELECT customer_unique_id, purchased_at AS second_ts
    FROM fct_orders
    WHERE is_valid = 1
      AND customer_order_seq = 2
      AND purchase_date < DATE '2018-09-01'
),
labelled AS (
    SELECT
        CAST(CAST(EXTRACT(YEAR FROM f.first_ts) AS INTEGER) AS TEXT) || '-Q'
            || CAST(CAST(EXTRACT(QUARTER FROM f.first_ts) AS INTEGER) AS TEXT)
            AS cohort_quarter,
        CASE WHEN s.second_ts IS NOT NULL THEN 1.0 ELSE 0.0 END AS is_repeat,
        CASE
            WHEN s.second_ts IS NOT NULL
                AND CAST(s.second_ts AS DATE) - CAST(f.first_ts AS DATE) <= 90 THEN 1.0
            ELSE 0.0
        END AS is_repeat_90d
    FROM firsts AS f
    LEFT JOIN seconds AS s ON s.customer_unique_id = f.customer_unique_id
)
SELECT cohort_quarter, customers, repeat_customers, repeat_rate, repeat_rate_90d
FROM (
    SELECT
        cohort_quarter,
        COUNT(*) AS customers,
        CAST(SUM(is_repeat) AS INTEGER) AS repeat_customers,
        AVG(is_repeat) AS repeat_rate,
        AVG(is_repeat_90d) AS repeat_rate_90d,
        0 AS sort_key
    FROM labelled
    GROUP BY cohort_quarter
    UNION ALL
    SELECT
        'All',
        COUNT(*),
        CAST(SUM(is_repeat) AS INTEGER),
        AVG(is_repeat),
        AVG(is_repeat_90d),
        1
    FROM labelled
) AS t
ORDER BY sort_key, cohort_quarter;

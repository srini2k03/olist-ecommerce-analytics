-- 10_repeat_purchase_rate.sql
-- Business question: What share of customers ever buy a second time, and is that
--   improving for newer cohorts?
-- Method: Per customer_unique_id, the first order time and the second order time
--   (ROW_NUMBER over purchase time). Cohort = quarter of the first order. A customer is
--   a repeater if a second order exists; repeat_90d also requires it within 90 days,
--   which compares cohorts fairly (later cohorts have had less time to come back).
--   A final 'All' row covers everyone.
-- Expected output: one row per quarter from 2017-Q1 to 2018-Q3 plus 'All'. Expect a
--   repeat rate of about 3% overall.
-- Dialect notes: the quarter label is built with CAST(... AS TEXT) || ..., which both
--   engines accept; the 90-day check uses DATE - DATE.
WITH customer_orders AS (
    SELECT
        c.customer_unique_id,
        o.order_purchase_timestamp,
        ROW_NUMBER() OVER (
            PARTITION BY c.customer_unique_id
            ORDER BY o.order_purchase_timestamp, o.order_id
        ) AS order_seq
    FROM orders AS o
    JOIN customers AS c ON c.customer_id = o.customer_id
    WHERE o.order_status NOT IN ('canceled', 'unavailable')
      AND o.order_purchase_timestamp >= DATE '2017-01-01'
      AND o.order_purchase_timestamp < DATE '2018-09-01'
),
per_customer AS (
    SELECT
        customer_unique_id,
        MAX(CASE WHEN order_seq = 1 THEN order_purchase_timestamp END) AS first_ts,
        MAX(CASE WHEN order_seq = 2 THEN order_purchase_timestamp END) AS second_ts
    FROM customer_orders
    GROUP BY customer_unique_id
),
labelled AS (
    SELECT
        CAST(CAST(EXTRACT(YEAR FROM first_ts) AS INTEGER) AS TEXT) || '-Q'
            || CAST(CAST(EXTRACT(QUARTER FROM first_ts) AS INTEGER) AS TEXT) AS cohort_quarter,
        CASE WHEN second_ts IS NOT NULL THEN 1.0 ELSE 0.0 END AS is_repeat,
        CASE
            WHEN second_ts IS NOT NULL
                AND CAST(second_ts AS DATE) - CAST(first_ts AS DATE) <= 90 THEN 1.0
            ELSE 0.0
        END AS is_repeat_90d
    FROM per_customer
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

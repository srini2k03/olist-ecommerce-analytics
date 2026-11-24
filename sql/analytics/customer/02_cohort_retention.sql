-- 02_cohort_retention.sql
-- Business question: Of the customers acquired in a given month, what share order again
--   in each later month?
-- Method: A customer's cohort is the month of their first valid order ever
--   (fct_orders.cohort_month), so a 2016 customer returning in 2017 is not counted as new.
--   Cohorts Jan 2017 to Aug 2018; activity = any valid order in the same window.
--   months_since_first comes from year/month arithmetic; cohort size is the month-0 count,
--   taken with a window MAX over each cohort.
-- Expected output: one row per (cohort_month, months_since_first) with active_customers,
--   cohort_size and retention (month 0 = 1.0). Month-1 retention is well under 1%.
-- Dialect notes: EXTRACT(...) is cast to INTEGER because Postgres returns NUMERIC.
WITH customer_months AS (
    SELECT DISTINCT
        customer_unique_id,
        cohort_month,
        purchase_month AS order_month
    FROM fct_orders
    WHERE is_valid = 1
      AND purchase_date >= DATE '2017-01-01'
      AND purchase_date < DATE '2018-09-01'
      AND cohort_month >= DATE '2017-01-01'
),
activity AS (
    SELECT
        cohort_month,
        customer_unique_id,
        CAST(
            (EXTRACT(YEAR FROM order_month) - EXTRACT(YEAR FROM cohort_month)) * 12
            + EXTRACT(MONTH FROM order_month) - EXTRACT(MONTH FROM cohort_month)
            AS INTEGER
        ) AS months_since_first
    FROM customer_months
),
counts AS (
    SELECT cohort_month, months_since_first, COUNT(*) AS active_customers
    FROM activity
    GROUP BY cohort_month, months_since_first
)
SELECT
    cohort_month,
    months_since_first,
    active_customers,
    MAX(CASE WHEN months_since_first = 0 THEN active_customers END)
        OVER (PARTITION BY cohort_month) AS cohort_size,
    1.0 * active_customers
        / MAX(CASE WHEN months_since_first = 0 THEN active_customers END)
            OVER (PARTITION BY cohort_month) AS retention
FROM counts
ORDER BY cohort_month, months_since_first;

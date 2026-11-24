-- 20_cohort_revenue.sql
-- Business question: How much GMV does an acquisition cohort generate after its first
--   month, and how much of a customer's value is the first order?
-- Method: Valid orders Jan 2017 to Aug 2018 for cohorts acquired in the same window
--   (fct_orders.cohort_month). Per cohort: size, GMV in month 0, GMV in later months,
--   GMV per customer and the share of cohort GMV that came after the first month.
-- Expected output: 20 cohort rows; repeat GMV is a few percent of first-month GMV.
-- Dialect notes: month arithmetic via EXTRACT, cast to INTEGER.
WITH orders_in_window AS (
    SELECT
        customer_unique_id,
        cohort_month,
        gmv,
        CAST(
            (EXTRACT(YEAR FROM purchase_month) - EXTRACT(YEAR FROM cohort_month)) * 12
            + EXTRACT(MONTH FROM purchase_month) - EXTRACT(MONTH FROM cohort_month)
            AS INTEGER
        ) AS months_since_first
    FROM fct_orders
    WHERE is_valid = 1
      AND purchase_date >= DATE '2017-01-01'
      AND purchase_date < DATE '2018-09-01'
      AND cohort_month >= DATE '2017-01-01'
)
SELECT
    cohort_month,
    COUNT(DISTINCT customer_unique_id) AS cohort_size,
    SUM(CASE WHEN months_since_first = 0 THEN gmv ELSE 0 END) AS gmv_month_0,
    SUM(CASE WHEN months_since_first > 0 THEN gmv ELSE 0 END) AS gmv_later_months,
    SUM(gmv) / COUNT(DISTINCT customer_unique_id) AS gmv_per_customer,
    SUM(CASE WHEN months_since_first > 0 THEN gmv ELSE 0 END) / SUM(gmv)
        AS share_gmv_after_month_0
FROM orders_in_window
GROUP BY cohort_month
ORDER BY cohort_month;

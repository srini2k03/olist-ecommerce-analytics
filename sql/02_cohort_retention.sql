-- 02_cohort_retention.sql
-- Business question: Of the customers acquired in a given month, what share come back
--   to buy again in each following month?
-- Method: Identify people with customer_unique_id (customer_id is per order). The cohort
--   is the month of the first valid order. For every later month with an order, compute
--   months_since_first. Retention = active customers / cohort size, where cohort size
--   comes from a window over month 0.
-- Expected output: long table (cohort_month, months_since_first, active_customers,
--   cohort_size, retention). Month 0 retention is 1.0 by construction; later months are
--   typically well under 1%.
-- Dialect notes: month arithmetic uses EXTRACT(YEAR/MONTH), which both engines support.
WITH customer_orders AS (
    SELECT
        c.customer_unique_id,
        CAST(date_trunc('month', o.order_purchase_timestamp) AS DATE) AS order_month
    FROM orders AS o
    JOIN customers AS c ON c.customer_id = o.customer_id
    WHERE o.order_status NOT IN ('canceled', 'unavailable')
      AND o.order_purchase_timestamp >= DATE '2017-01-01'
      AND o.order_purchase_timestamp < DATE '2018-09-01'
),
first_month AS (
    SELECT customer_unique_id, MIN(order_month) AS cohort_month
    FROM customer_orders
    GROUP BY customer_unique_id
),
activity AS (
    SELECT DISTINCT
        f.cohort_month,
        co.customer_unique_id,
        CAST(
            (EXTRACT(YEAR FROM co.order_month) - EXTRACT(YEAR FROM f.cohort_month)) * 12
            + EXTRACT(MONTH FROM co.order_month) - EXTRACT(MONTH FROM f.cohort_month)
            AS INTEGER
        ) AS months_since_first
    FROM customer_orders AS co
    JOIN first_month AS f ON f.customer_unique_id = co.customer_unique_id
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

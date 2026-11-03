-- 09_state_delivery_sla.sql
-- Business question: Which customer states get slow or unreliable delivery, and is the
--   promised date padded enough to absorb it?
-- Method: Delivered orders by customer_state. Delivery days = delivered date - purchase
--   date; promised days = estimated date - purchase date. We report the mean and p90 of
--   delivery days, the mean promised days, the late rate (delivered after the estimated
--   date) and the mean review. RANK() by late rate; states need at least 300 orders.
-- Expected output: about 20-27 rows, worst late rate first. Expect the north/north-east
--   (e.g. AL, MA) to be slowest and SP the fastest.
-- Dialect notes: DATE - DATE for whole days; PERCENTILE_CONT for p90.
WITH order_review AS (
    SELECT order_id, AVG(review_score) AS review_score
    FROM order_reviews
    GROUP BY order_id
),
delivered AS (
    SELECT
        c.customer_state,
        CAST(o.order_delivered_customer_date AS DATE)
            - CAST(o.order_purchase_timestamp AS DATE) AS delivery_days,
        CAST(o.order_estimated_delivery_date AS DATE)
            - CAST(o.order_purchase_timestamp AS DATE) AS promised_days,
        CASE
            WHEN CAST(o.order_delivered_customer_date AS DATE)
                > CAST(o.order_estimated_delivery_date AS DATE) THEN 1.0
            ELSE 0.0
        END AS is_late,
        r.review_score
    FROM orders AS o
    JOIN customers AS c ON c.customer_id = o.customer_id
    LEFT JOIN order_review AS r ON r.order_id = o.order_id
    WHERE o.order_status = 'delivered'
      AND o.order_delivered_customer_date IS NOT NULL
      AND o.order_estimated_delivery_date IS NOT NULL
),
by_state AS (
    SELECT
        customer_state,
        COUNT(*) AS orders,
        AVG(1.0 * delivery_days) AS avg_delivery_days,
        PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY delivery_days) AS p90_delivery_days,
        AVG(1.0 * promised_days) AS avg_promised_days,
        AVG(is_late) AS late_rate,
        AVG(review_score) AS avg_review
    FROM delivered
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

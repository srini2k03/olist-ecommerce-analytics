-- 04_delivery_delay_vs_review.sql
-- Business question: How much does delivering late (relative to the date promised at
--   checkout) cost us in review score?
-- Method: Delivered orders with both a delivery date and an estimated date.
--   delay_days = delivered date - estimated date (positive = late). Orders with several
--   reviews use their mean score. Orders go into delay buckets; per bucket we report
--   volume, mean score and the share of 1-star and 5-star reviews.
-- Expected output: 7 rows ordered from very early to very late. Mean score should fall
--   sharply once delay_days > 0.
-- Dialect notes: DATE - DATE for whole days; the bucket order is a numeric sort key.
WITH order_review AS (
    SELECT order_id, AVG(review_score) AS review_score
    FROM order_reviews
    GROUP BY order_id
),
delivered AS (
    SELECT
        o.order_id,
        CAST(o.order_delivered_customer_date AS DATE)
            - CAST(o.order_estimated_delivery_date AS DATE) AS delay_days,
        r.review_score
    FROM orders AS o
    JOIN order_review AS r ON r.order_id = o.order_id
    WHERE o.order_status = 'delivered'
      AND o.order_delivered_customer_date IS NOT NULL
      AND o.order_estimated_delivery_date IS NOT NULL
),
bucketed AS (
    SELECT
        *,
        CASE
            WHEN delay_days <= -15 THEN 1
            WHEN delay_days <= -8 THEN 2
            WHEN delay_days <= 0 THEN 3
            WHEN delay_days <= 3 THEN 4
            WHEN delay_days <= 7 THEN 5
            WHEN delay_days <= 14 THEN 6
            ELSE 7
        END AS bucket_order
    FROM delivered
)
SELECT
    bucket_order,
    CASE bucket_order
        WHEN 1 THEN '15+ days early'
        WHEN 2 THEN '8-14 days early'
        WHEN 3 THEN '0-7 days early'
        WHEN 4 THEN '1-3 days late'
        WHEN 5 THEN '4-7 days late'
        WHEN 6 THEN '8-14 days late'
        ELSE '15+ days late'
    END AS delay_bucket,
    COUNT(*) AS orders,
    1.0 * COUNT(*) / SUM(COUNT(*)) OVER () AS order_share,
    AVG(review_score) AS avg_review,
    AVG(CASE WHEN review_score <= 1 THEN 1.0 ELSE 0.0 END) AS share_1_star,
    AVG(CASE WHEN review_score >= 5 THEN 1.0 ELSE 0.0 END) AS share_5_star
FROM bucketed
GROUP BY bucket_order
ORDER BY bucket_order;

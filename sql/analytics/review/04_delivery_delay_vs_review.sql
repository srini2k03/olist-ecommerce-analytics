-- 04_delivery_delay_vs_review.sql
-- Business question: How much does delivering late (relative to the date promised at
--   checkout) cost us in review score?
-- Method: Delivered orders (fct_orders.is_delivered) with a review. delay_days = delivered
--   date - estimated date (positive = late). Orders with several reviews use their mean
--   score (computed in fct_orders). Orders go into delay buckets; per bucket we report
--   volume, mean score and the share of 1-star and 5-star reviews.
-- Expected output: 7 rows ordered from very early to very late. Mean score should fall
--   sharply once delay_days > 0.
-- Dialect notes: the bucket order is a numeric sort key.
WITH bucketed AS (
    SELECT
        review_score,
        CASE
            WHEN delay_days <= -15 THEN 1
            WHEN delay_days <= -8 THEN 2
            WHEN delay_days <= 0 THEN 3
            WHEN delay_days <= 3 THEN 4
            WHEN delay_days <= 7 THEN 5
            WHEN delay_days <= 14 THEN 6
            ELSE 7
        END AS bucket_order
    FROM fct_orders
    WHERE is_delivered = 1
      AND review_score IS NOT NULL
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

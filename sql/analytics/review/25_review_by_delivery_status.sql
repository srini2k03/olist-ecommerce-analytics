-- 25_review_by_delivery_status.sql
-- Business question: What is the average review of orders delivered on time, delivered
--   late and not delivered? (Verifies the headline late-vs-on-time finding directly.)
-- Method: Valid orders with a review from fct_orders. Status: 'late' = delivered after the
--   estimated date, 'on time' = delivered on or before it, 'not delivered' = no delivery
--   (cancelled orders are excluded by is_valid). Mean score, share of 1-star and 5-star.
-- Expected output: 3 rows; late orders average about 2 points below on-time orders.
-- Dialect notes: none beyond CASE labels.
SELECT
    CASE
        WHEN is_delivered = 1 AND is_late = 1 THEN 'late'
        WHEN is_delivered = 1 THEN 'on time'
        ELSE 'not delivered'
    END AS delivery_status,
    COUNT(*) AS orders,
    AVG(review_score) AS avg_review,
    AVG(CASE WHEN review_score <= 1 THEN 1.0 ELSE 0.0 END) AS share_1_star,
    AVG(CASE WHEN review_score >= 5 THEN 1.0 ELSE 0.0 END) AS share_5_star
FROM fct_orders
WHERE is_valid = 1
  AND review_score IS NOT NULL
GROUP BY 1
ORDER BY 1;

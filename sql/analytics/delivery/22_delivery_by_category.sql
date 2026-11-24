-- 22_delivery_by_category.sql
-- Business question: Which product categories are delivered slowly or late, and do their
--   reviews suffer?
-- Method: Delivered valid orders from fct_orders by main category (the category of item
--   1; 727 orders span several categories). Mean and p90 delivery days, mean promised
--   days, late rate, mean freight, mean weight of the main product and mean review.
--   Categories need at least 500 delivered orders. RANK by late rate.
-- Expected output: about 30 categories, worst late rate first.
-- Dialect notes: PERCENTILE_CONT for p90.
SELECT
    RANK() OVER (ORDER BY AVG(1.0 * is_late) DESC) AS late_rank,
    category,
    COUNT(*) AS delivered_orders,
    AVG(1.0 * delivery_days) AS avg_delivery_days,
    PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY delivery_days) AS p90_delivery_days,
    AVG(1.0 * estimated_days) AS avg_promised_days,
    AVG(1.0 * is_late) AS late_rate,
    AVG(freight_value) AS avg_freight,
    AVG(1.0 * main_product_weight_g) AS avg_weight_g,
    AVG(review_score) AS avg_review
FROM fct_orders
WHERE is_delivered = 1
  AND is_valid = 1
GROUP BY category
HAVING COUNT(*) >= 500
ORDER BY late_rank, category;

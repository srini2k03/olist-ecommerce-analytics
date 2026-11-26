-- 22_delivery_by_category.sql (BigQuery version)
-- Same question, method and output as sql/analytics/delivery/22_delivery_by_category.sql.
-- Dialect notes: exact percentile via PERCENTILE_CONT(x, p) OVER (PARTITION BY category) on
--   the filtered rows, carried through the GROUP BY with ANY_VALUE.
WITH base AS (
    SELECT
        category,
        is_late,
        delivery_days,
        estimated_days,
        freight_value,
        main_product_weight_g,
        review_score,
        PERCENTILE_CONT(delivery_days, 0.9) OVER (PARTITION BY category) AS p90
    FROM fct_orders
    WHERE is_delivered = 1
      AND is_valid = 1
)
SELECT
    RANK() OVER (ORDER BY AVG(1.0 * is_late) DESC) AS late_rank,
    category,
    COUNT(*) AS delivered_orders,
    AVG(1.0 * delivery_days) AS avg_delivery_days,
    ANY_VALUE(p90) AS p90_delivery_days,
    AVG(1.0 * estimated_days) AS avg_promised_days,
    AVG(1.0 * is_late) AS late_rate,
    AVG(freight_value) AS avg_freight,
    AVG(1.0 * main_product_weight_g) AS avg_weight_g,
    AVG(review_score) AS avg_review
FROM base
GROUP BY category
HAVING COUNT(*) >= 500
ORDER BY late_rank, category;

-- 08_freight_ratio.sql (BigQuery version)
-- Same question, method and output as sql/analytics/product/08_freight_ratio.sql.
-- Dialect notes: exact median via PERCENTILE_CONT(x, 0.5) OVER (PARTITION BY category),
--   carried through the GROUP BY with ANY_VALUE.
WITH items AS (
    SELECT
        category,
        item_revenue AS price,
        freight_value,
        freight_value / NULLIF(item_revenue, 0) AS freight_ratio,
        weight_g,
        review_score
    FROM fct_order_items
    WHERE is_valid = 1
),
items_with_median AS (
    SELECT
        *,
        PERCENTILE_CONT(freight_ratio, 0.5) OVER (PARTITION BY category) AS median_ratio
    FROM items
),
by_category AS (
    SELECT
        category,
        COUNT(*) AS items,
        AVG(price) AS avg_price,
        AVG(freight_value) AS avg_freight,
        ANY_VALUE(median_ratio) AS median_freight_ratio,
        AVG(CASE WHEN freight_ratio >= 0.5 THEN 1.0 ELSE 0.0 END) AS share_freight_ge_50pct,
        AVG(1.0 * weight_g) AS avg_weight_g,
        AVG(review_score) AS avg_review
    FROM items_with_median
    GROUP BY category
    HAVING COUNT(*) >= 1000
)
SELECT
    DENSE_RANK() OVER (ORDER BY median_freight_ratio DESC) AS freight_burden_rank,
    *
FROM by_category
ORDER BY freight_burden_rank, category;

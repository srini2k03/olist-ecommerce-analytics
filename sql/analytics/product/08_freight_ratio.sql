-- 08_freight_ratio.sql
-- Business question: In which categories does shipping cost eat most of the item price,
--   and does that hurt reviews?
-- Method: Items of valid orders from fct_order_items. freight_ratio = freight / price per
--   item. Per category (at least 1,000 items): mean price and freight, median ratio, share
--   of items where freight is at least half the price, mean weight and mean review.
--   DENSE_RANK by median ratio.
-- Expected output: about 30 categories, the heaviest freight burden first.
-- Dialect notes: NULLIF guards the division; PERCENTILE_CONT for the median.
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
by_category AS (
    SELECT
        category,
        COUNT(*) AS items,
        AVG(price) AS avg_price,
        AVG(freight_value) AS avg_freight,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY freight_ratio) AS median_freight_ratio,
        AVG(CASE WHEN freight_ratio >= 0.5 THEN 1.0 ELSE 0.0 END) AS share_freight_ge_50pct,
        AVG(1.0 * weight_g) AS avg_weight_g,
        AVG(review_score) AS avg_review
    FROM items
    GROUP BY category
    HAVING COUNT(*) >= 1000
)
SELECT
    DENSE_RANK() OVER (ORDER BY median_freight_ratio DESC) AS freight_burden_rank,
    *
FROM by_category
ORDER BY freight_burden_rank, category;

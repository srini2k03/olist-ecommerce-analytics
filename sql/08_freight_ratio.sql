-- 08_freight_ratio.sql
-- Business question: In which categories is shipping expensive relative to the product,
--   and do customers there review us worse?
-- Method: Item-level freight_value / price for categories with at least 1,000 sold
--   items. Median ratio via PERCENTILE_CONT (robust to cheap items with high freight),
--   share of items where freight is at least 50% of the price, mean product weight and
--   the mean review score of orders containing the category. DENSE_RANK by median ratio.
-- Expected output: about 30-40 rows, highest freight burden first.
-- Dialect notes: PERCENTILE_CONT ... WITHIN GROUP works in both engines.
WITH order_review AS (
    SELECT order_id, AVG(review_score) AS review_score
    FROM order_reviews
    GROUP BY order_id
),
items AS (
    SELECT
        COALESCE(t.product_category_name_english, p.product_category_name, 'unknown')
            AS category,
        oi.price,
        oi.freight_value,
        oi.freight_value / NULLIF(oi.price, 0) AS freight_ratio,
        p.product_weight_g,
        r.review_score
    FROM order_items AS oi
    JOIN orders AS o ON o.order_id = oi.order_id
    JOIN products AS p ON p.product_id = oi.product_id
    LEFT JOIN category_translation AS t
        ON t.product_category_name = p.product_category_name
    LEFT JOIN order_review AS r ON r.order_id = oi.order_id
    WHERE o.order_status NOT IN ('canceled', 'unavailable')
),
by_category AS (
    SELECT
        category,
        COUNT(*) AS items,
        AVG(price) AS avg_price,
        AVG(freight_value) AS avg_freight,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY freight_ratio) AS median_freight_ratio,
        AVG(CASE WHEN freight_ratio >= 0.5 THEN 1.0 ELSE 0.0 END) AS share_freight_ge_50pct,
        AVG(1.0 * product_weight_g) AS avg_weight_g,
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

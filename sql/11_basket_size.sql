-- 11_basket_size.sql
-- Business question: How many items do customers put in an order, and how do order
--   value, freight per item and satisfaction change with basket size?
-- Method: Count items per valid order (one row per unit in order_items), bucket into
--   1, 2, 3 and 4+, and report volume, mean order value, mean freight per item and
--   mean review score. Multi-item baskets are the lever a free-shipping threshold
--   would pull, so this sets the baseline for the simulated A/B test in 02_stats.
-- Expected output: 4 rows. Single-item orders should be about 90% of volume.
-- Dialect notes: none needed; only standard aggregates and window functions.
WITH order_review AS (
    SELECT order_id, AVG(review_score) AS review_score
    FROM order_reviews
    GROUP BY order_id
),
baskets AS (
    SELECT
        oi.order_id,
        COUNT(*) AS items,
        COUNT(DISTINCT oi.product_id) AS distinct_products,
        SUM(oi.price + oi.freight_value) AS order_value,
        SUM(oi.freight_value) / COUNT(*) AS freight_per_item
    FROM order_items AS oi
    JOIN orders AS o ON o.order_id = oi.order_id
    WHERE o.order_status NOT IN ('canceled', 'unavailable')
    GROUP BY oi.order_id
)
SELECT
    CASE WHEN b.items >= 4 THEN '4+' ELSE CAST(b.items AS TEXT) END AS basket_size,
    COUNT(*) AS orders,
    1.0 * COUNT(*) / SUM(COUNT(*)) OVER () AS order_share,
    AVG(b.order_value) AS avg_order_value,
    AVG(b.freight_per_item) AS avg_freight_per_item,
    AVG(1.0 * b.distinct_products) AS avg_distinct_products,
    AVG(r.review_score) AS avg_review
FROM baskets AS b
LEFT JOIN order_review AS r ON r.order_id = b.order_id
GROUP BY 1
ORDER BY 1;

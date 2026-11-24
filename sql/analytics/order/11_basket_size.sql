-- 11_basket_size.sql
-- Business question: Do bigger baskets mean higher value and happier customers?
-- Method: Valid orders from fct_orders grouped by number of items (4+ pooled): share of
--   orders, mean GMV, mean freight per item, mean distinct products and mean review.
-- Expected output: 4 rows ('1', '2', '3', '4+'); order shares sum to 1.
-- Dialect notes: CAST(n AS TEXT) for the label on both engines.
SELECT
    CASE WHEN n_items >= 4 THEN '4+' ELSE CAST(n_items AS TEXT) END AS basket_size,
    COUNT(*) AS orders,
    1.0 * COUNT(*) / SUM(COUNT(*)) OVER () AS order_share,
    AVG(gmv) AS avg_order_value,
    AVG(freight_value / n_items) AS avg_freight_per_item,
    AVG(1.0 * n_products) AS avg_distinct_products,
    AVG(review_score) AS avg_review
FROM fct_orders
WHERE is_valid = 1
GROUP BY 1
ORDER BY 1;

-- 24_category_scorecard.sql
-- Business question: For each category, how big is it, how popular, how satisfied are its
--   buyers and how reliably is it delivered?
-- Method: Item-level sales from fct_order_items (valid orders, full history): item revenue,
--   items, orders, distinct buyers, mean price, freight share of GMV; order-level mean
--   review, 1-2 star share and late rate over the orders containing the category.
-- Expected output: one row per category (about 70), largest revenue first.
-- Dialect notes: per-order flags are taken once per (category, order) before averaging.
WITH category_orders AS (
    SELECT
        category,
        order_id,
        MAX(customer_unique_id) AS customer_unique_id,
        SUM(item_revenue) AS item_revenue,
        SUM(freight_value) AS freight_value,
        COUNT(*) AS items,
        MAX(review_score) AS review_score,
        MAX(is_late) AS is_late
    FROM fct_order_items
    WHERE is_valid = 1
    GROUP BY category, order_id
)
SELECT
    category,
    SUM(item_revenue) AS item_revenue,
    SUM(items) AS items,
    COUNT(*) AS orders,
    COUNT(DISTINCT customer_unique_id) AS buyers,
    SUM(item_revenue) / SUM(items) AS avg_item_price,
    SUM(freight_value) / (SUM(item_revenue) + SUM(freight_value)) AS freight_share_of_gmv,
    AVG(review_score) AS avg_review,
    AVG(CASE WHEN review_score <= 2 THEN 1.0 WHEN review_score IS NOT NULL THEN 0.0 END)
        AS bad_review_rate,
    AVG(1.0 * is_late) AS late_rate
FROM category_orders
GROUP BY category
ORDER BY item_revenue DESC, category;

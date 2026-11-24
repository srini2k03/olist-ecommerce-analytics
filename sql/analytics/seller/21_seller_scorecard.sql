-- 21_seller_scorecard.sql
-- Business question: How does every seller perform on volume, money, speed, punctuality,
--   satisfaction and cancellations? (Input to the uncertainty-aware ranking.)
-- Method: Per (seller, order) from fct_order_items, then per seller: orders placed
--   (including cancelled), valid orders, item revenue, delivered orders, late orders,
--   mean delivery days, reviewed orders, mean review, 1-2 star reviews, cancelled orders.
--   Rates are left to the caller so it can apply minimum volumes and shrinkage.
-- Expected output: one row per seller that ever received an order (about 3,000).
-- Dialect notes: MAX over the per-order flags keeps one vote per (seller, order).
WITH seller_orders AS (
    SELECT
        seller_id,
        order_id,
        MAX(seller_state) AS seller_state,
        MAX(is_valid) AS is_valid,
        MAX(is_cancelled) AS is_cancelled,
        MAX(is_delivered) AS is_delivered,
        MAX(is_late) AS is_late,
        MAX(delivery_days) AS delivery_days,
        MAX(review_score) AS review_score,
        SUM(CASE WHEN is_valid = 1 THEN item_revenue ELSE 0 END) AS item_revenue
    FROM fct_order_items
    GROUP BY seller_id, order_id
)
SELECT
    seller_id,
    MAX(seller_state) AS seller_state,
    COUNT(*) AS orders_placed,
    SUM(is_valid) AS orders,
    SUM(item_revenue) AS item_revenue,
    SUM(is_delivered) AS delivered_orders,
    SUM(CASE WHEN is_delivered = 1 THEN is_late ELSE 0 END) AS late_orders,
    AVG(CASE WHEN is_delivered = 1 THEN 1.0 * delivery_days END) AS avg_delivery_days,
    SUM(CASE WHEN review_score IS NOT NULL AND is_valid = 1 THEN 1 ELSE 0 END)
        AS reviewed_orders,
    AVG(CASE WHEN is_valid = 1 THEN review_score END) AS avg_review,
    SUM(CASE WHEN review_score <= 2 AND is_valid = 1 THEN 1 ELSE 0 END) AS bad_reviews,
    SUM(is_cancelled) AS cancelled_orders
FROM seller_orders
GROUP BY seller_id
ORDER BY item_revenue DESC, seller_id;

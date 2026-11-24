-- fct_order_items
-- Layer: marts
-- Grain: one row per (order_id, order_item_id).
-- Purpose: item-level sales for category and seller analytics. An order can span several
--   sellers (1,278 orders) and categories (727 orders), so seller and category revenue
--   must be summed here, not from fct_orders' main seller/category.
-- Depends on: stg_order_items, stg_products, stg_sellers, fct_orders
-- Money: item_revenue = price; freight_value; gmv = price + freight_value.
SELECT
    i.order_id,
    i.order_item_id,
    i.product_id,
    i.seller_id,
    p.category,
    s.state AS seller_state,
    o.customer_unique_id,
    o.customer_state,
    o.order_status,
    o.is_valid,
    o.is_cancelled,
    o.is_delivered,
    o.purchased_at,
    o.purchase_month,
    o.delivery_days,
    o.delay_days,
    o.is_late,
    o.review_score,
    i.price AS item_revenue,
    i.freight_value,
    i.price + i.freight_value AS gmv,
    p.weight_g,
    p.volume_cm3
FROM stg_order_items AS i
JOIN fct_orders AS o ON o.order_id = i.order_id
LEFT JOIN stg_products AS p ON p.product_id = i.product_id
LEFT JOIN stg_sellers AS s ON s.seller_id = i.seller_id

-- stg_order_items
-- Layer: staging
-- Grain: one row per (order_id, order_item_id): one row per unit sold.
-- Purpose: line items with price and freight kept separate.
-- Depends on: order_items
-- Transformations: none beyond renaming; price is the item price paid to the seller and
--   freight_value the shipping charged for that item (Olist splits an order's freight
--   across its items).
SELECT
    order_id,
    order_item_id,
    product_id,
    seller_id,
    shipping_limit_date AS shipping_limit_at,
    price,
    freight_value
FROM order_items

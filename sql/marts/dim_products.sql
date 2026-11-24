-- dim_products
-- Layer: marts
-- Grain: one row per product_id.
-- Purpose: the product dimension analytics read (category in English, size, weight).
-- Depends on: stg_products
SELECT
    product_id,
    category,
    category_pt,
    name_length,
    description_length,
    photos_qty,
    weight_g,
    length_cm,
    height_cm,
    width_cm,
    volume_cm3
FROM stg_products

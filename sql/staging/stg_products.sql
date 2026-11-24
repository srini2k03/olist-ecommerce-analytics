-- stg_products
-- Layer: staging
-- Grain: one row per product_id.
-- Purpose: products with an English category and physical attributes.
-- Depends on: products, category_translation
-- Transformations: category translated to English; the two categories missing from the
--   translation table are translated here; the 610 products without a category get
--   'unknown'. The source column names are misspelt (lenght) and are renamed. Volume is
--   length x height x width.
SELECT
    p.product_id,
    p.product_category_name AS category_pt,
    COALESCE(
        t.product_category_name_english,
        CASE p.product_category_name
            WHEN 'pc_gamer' THEN 'pc_gamer'
            WHEN 'portateis_cozinha_e_preparadores_de_alimentos'
                THEN 'portable_kitchen_food_preparers'
        END,
        p.product_category_name,
        'unknown'
    ) AS category,
    p.product_name_lenght AS name_length,
    p.product_description_lenght AS description_length,
    p.product_photos_qty AS photos_qty,
    p.product_weight_g AS weight_g,
    p.product_length_cm AS length_cm,
    p.product_height_cm AS height_cm,
    p.product_width_cm AS width_cm,
    p.product_length_cm * p.product_height_cm * p.product_width_cm AS volume_cm3
FROM products AS p
LEFT JOIN category_translation AS t
    ON t.product_category_name = p.product_category_name

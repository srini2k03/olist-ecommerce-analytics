-- dim_sellers
-- Layer: marts
-- Grain: one row per seller_id.
-- Purpose: seller location (with coordinates when the zip prefix is known) and first sale.
-- Depends on: stg_sellers, stg_geolocation, fct_order_items
SELECT
    s.seller_id,
    s.zip_prefix,
    s.city,
    s.state,
    g.lat,
    g.lng,
    MIN(i.purchased_at) AS first_sale_at
FROM stg_sellers AS s
LEFT JOIN stg_geolocation AS g ON g.zip_prefix = s.zip_prefix
LEFT JOIN fct_order_items AS i ON i.seller_id = s.seller_id AND i.is_valid = 1
GROUP BY s.seller_id, s.zip_prefix, s.city, s.state, g.lat, g.lng

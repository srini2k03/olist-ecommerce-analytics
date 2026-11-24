-- stg_geolocation
-- Layer: staging
-- Grain: one row per 5-digit zip prefix.
-- Purpose: one coordinate per zip prefix, for distances between seller and customer.
-- Depends on: geolocation
-- Transformations: the raw table has 1,000,163 rows for 19,015 prefixes (many duplicates
--   and a few points outside Brazil). Points outside Brazil's bounding box are dropped and
--   the remaining points are reduced to their median latitude and longitude.
SELECT
    LPAD(TRIM(geolocation_zip_code_prefix), 5, '0') AS zip_prefix,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY geolocation_lat) AS lat,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY geolocation_lng) AS lng,
    MIN(UPPER(TRIM(geolocation_state))) AS state,
    COUNT(*) AS n_points
FROM geolocation
WHERE geolocation_lat BETWEEN -33.8 AND 5.3
  AND geolocation_lng BETWEEN -74.0 AND -34.7
GROUP BY LPAD(TRIM(geolocation_zip_code_prefix), 5, '0')

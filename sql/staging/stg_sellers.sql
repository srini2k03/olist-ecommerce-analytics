-- stg_sellers
-- Layer: staging
-- Grain: one row per seller_id.
-- Purpose: normalised seller attributes.
-- Depends on: sellers
-- Transformations: zip prefix left-padded to 5 digits (1,027 sellers had lost the leading
--   zero), city lower-cased and trimmed, state upper-cased and trimmed.
SELECT
    seller_id,
    LPAD(TRIM(seller_zip_code_prefix), 5, '0') AS zip_prefix,
    LOWER(TRIM(seller_city)) AS city,
    UPPER(TRIM(seller_state)) AS state
FROM sellers

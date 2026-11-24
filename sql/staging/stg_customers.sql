-- stg_customers
-- Layer: staging
-- Grain: one row per customer_id (Olist issues a new customer_id for every order;
--   customer_unique_id identifies the person).
-- Purpose: normalised customer attributes.
-- Depends on: customers
-- Transformations: zip prefixes are left-padded to 5 digits (the CSV dropped the leading
--   zero of 23,995 customers, so they never matched the geolocation table); city lower-cased
--   and trimmed; state upper-cased and trimmed.
SELECT
    customer_id,
    customer_unique_id,
    LPAD(TRIM(customer_zip_code_prefix), 5, '0') AS zip_prefix,
    LOWER(TRIM(customer_city)) AS city,
    UPPER(TRIM(customer_state)) AS state
FROM customers

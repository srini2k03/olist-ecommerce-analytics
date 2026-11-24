-- stg_order_payments
-- Layer: staging
-- Grain: one row per (order_id, payment_sequential).
-- Purpose: payments as recorded by the payment provider.
-- Depends on: order_payments
-- Transformations: payment_type lower-cased; 2 rows with 0 instalments are set to 1 (a
--   payment is at least one instalment). 'not_defined' (3 rows, value 0) is kept.
SELECT
    order_id,
    payment_sequential,
    LOWER(TRIM(payment_type)) AS payment_type,
    CASE WHEN payment_installments < 1 THEN 1 ELSE payment_installments END AS installments,
    payment_value
FROM order_payments

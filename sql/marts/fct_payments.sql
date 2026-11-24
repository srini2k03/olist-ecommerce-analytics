-- fct_payments
-- Layer: marts
-- Grain: one row per (order_id, payment_sequential).
-- Purpose: payment-method analytics with the order's validity flag and purchase month.
-- Depends on: stg_order_payments, fct_orders
-- Money: payment_value is cash charged to the customer (may include instalment interest);
--   it is not revenue and not GMV.
SELECT
    p.order_id,
    p.payment_sequential,
    p.payment_type,
    p.installments,
    p.payment_value,
    o.is_valid,
    o.is_cancelled,
    o.purchase_month,
    o.customer_state
FROM stg_order_payments AS p
JOIN fct_orders AS o ON o.order_id = p.order_id

-- 07_payment_mix.sql
-- Business question: How do customers pay, and how much do they rely on instalments?
-- Method: From order_payments for valid orders: per payment_type, the number of orders
--   using it, the share of total payment value, mean payment value and mean instalments,
--   and the share of payments split into more than one instalment. An order can use
--   several types (e.g. a voucher plus a card), so order shares can add up to over 100%.
-- Expected output: one row per payment type (credit_card, boleto, voucher, debit_card,
--   and possibly not_defined), ordered by value share.
-- Dialect notes: COUNT(DISTINCT ...) together with window totals is portable.
WITH valid_payments AS (
    SELECT p.*
    FROM order_payments AS p
    JOIN orders AS o ON o.order_id = p.order_id
    WHERE o.order_status NOT IN ('canceled', 'unavailable')
),
totals AS (
    SELECT COUNT(DISTINCT order_id) AS all_orders, SUM(payment_value) AS all_value
    FROM valid_payments
)
SELECT
    vp.payment_type,
    COUNT(DISTINCT vp.order_id) AS orders,
    1.0 * COUNT(DISTINCT vp.order_id) / MAX(t.all_orders) AS order_share,
    SUM(vp.payment_value) / MAX(t.all_value) AS value_share,
    AVG(vp.payment_value) AS avg_payment_value,
    AVG(1.0 * vp.payment_installments) AS avg_installments,
    AVG(CASE WHEN vp.payment_installments > 1 THEN 1.0 ELSE 0.0 END) AS share_multi_installment
FROM valid_payments AS vp
CROSS JOIN totals AS t
GROUP BY vp.payment_type
ORDER BY value_share DESC;

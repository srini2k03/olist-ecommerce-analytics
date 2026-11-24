-- 07_payment_mix.sql
-- Business question: How do customers pay, and how much do they rely on instalments?
-- Method: Payments of valid orders from fct_payments. Per payment type: orders using it,
--   share of orders and of payment value, mean payment value, mean instalments and the
--   share paid in more than one instalment. Payment value is cash charged, not revenue.
-- Expected output: one row per payment type (credit_card, boleto, voucher, debit_card,
--   possibly not_defined), largest value share first.
-- Dialect notes: COUNT(DISTINCT ...) and a CROSS JOIN to a totals row.
WITH valid_payments AS (
    SELECT *
    FROM fct_payments
    WHERE is_valid = 1
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
    AVG(1.0 * vp.installments) AS avg_installments,
    AVG(CASE WHEN vp.installments > 1 THEN 1.0 ELSE 0.0 END) AS share_multi_installment
FROM valid_payments AS vp
CROSS JOIN totals AS t
GROUP BY vp.payment_type
ORDER BY value_share DESC;

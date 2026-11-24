-- 18_money_reconciliation.sql
-- Business question: How do item revenue, freight, GMV and payment value relate, and how
--   far apart are they? (They are different quantities and must not be used
--   interchangeably.)
-- Method: Valid orders from fct_orders by purchase month (Jan 2017 to Aug 2018):
--   item_revenue (prices), freight_value, gmv = item_revenue + freight, payment_value
--   (cash charged, incl. instalment interest and vouchers), the payment-GMV gap, freight's
--   share of GMV and the number of orders whose payments differ from GMV by more than R$ 1.
-- Expected output: 20 monthly rows. Freight is roughly 14% of GMV; payment value is within
--   a fraction of a percent of GMV in total, but a few hundred orders differ.
-- Dialect notes: ABS and CASE are portable.
SELECT
    purchase_month AS month,
    COUNT(*) AS orders,
    SUM(item_revenue) AS item_revenue,
    SUM(freight_value) AS freight_value,
    SUM(gmv) AS gmv,
    SUM(payment_value) AS payment_value,
    SUM(payment_value) - SUM(gmv) AS payment_minus_gmv,
    SUM(freight_value) / SUM(gmv) AS freight_share_of_gmv,
    SUM(CASE WHEN ABS(payment_minus_gmv) > 1 THEN 1 ELSE 0 END) AS orders_payment_differs
FROM fct_orders
WHERE is_valid = 1
  AND purchase_date >= DATE '2017-01-01'
  AND purchase_date < DATE '2018-09-01'
GROUP BY purchase_month
ORDER BY purchase_month;

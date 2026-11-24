-- 17_order_status_funnel.sql
-- Business question: What share of orders are cancelled, and what share of placed orders
--   are actually delivered, month by month?
-- Method: Every order in fct_orders (including cancelled ones) by purchase month, Jan 2017
--   to Aug 2018. cancellation_rate = cancelled or unavailable / placed;
--   delivery_completion_rate = delivered (status delivered with a date) / placed;
--   in_progress = neither cancelled nor delivered at extract time (shipped, invoiced...).
-- Expected output: 20 monthly rows; cancellation rate around 1%, completion around 97%.
-- Dialect notes: integer flags summed and divided with 1.0 * to avoid integer division.
SELECT
    purchase_month AS month,
    COUNT(*) AS orders_placed,
    SUM(is_cancelled) AS orders_cancelled,
    SUM(is_delivered) AS orders_delivered,
    COUNT(*) - SUM(is_cancelled) - SUM(is_delivered) AS orders_in_progress,
    1.0 * SUM(is_cancelled) / COUNT(*) AS cancellation_rate,
    1.0 * SUM(is_delivered) / COUNT(*) AS delivery_completion_rate
FROM fct_orders
WHERE purchase_date >= DATE '2017-01-01'
  AND purchase_date < DATE '2018-09-01'
GROUP BY purchase_month
ORDER BY purchase_month;

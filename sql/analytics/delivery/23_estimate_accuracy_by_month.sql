-- 23_estimate_accuracy_by_month.sql
-- Business question: How good is Olist's promised delivery date, and how has it changed?
--   (Promised vs actual, per purchase month.)
-- Method: Delivered valid orders purchased Jan 2017 to Aug 2018. Per purchase month: mean
--   promised days (estimated - purchase), mean and median actual days, mean buffer
--   (promised - actual), late rate, and the mean absolute error of the promise in days.
-- Expected output: 20 monthly rows. Promises are padded (actual well below promised on
--   average) yet late rates spike in some months (e.g. Nov 2017, Feb-Mar 2018).
-- Dialect notes: ABS over integer day counts; PERCENTILE_CONT for the median.
SELECT
    purchase_month AS month,
    COUNT(*) AS delivered_orders,
    AVG(1.0 * estimated_days) AS avg_promised_days,
    AVG(1.0 * delivery_days) AS avg_actual_days,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY delivery_days) AS median_actual_days,
    AVG(1.0 * (estimated_days - delivery_days)) AS avg_buffer_days,
    AVG(1.0 * is_late) AS late_rate,
    AVG(1.0 * ABS(estimated_days - delivery_days)) AS promise_mae_days
FROM fct_orders
WHERE is_delivered = 1
  AND is_valid = 1
  AND purchase_date >= DATE '2017-01-01'
  AND purchase_date < DATE '2018-09-01'
GROUP BY purchase_month
ORDER BY purchase_month;

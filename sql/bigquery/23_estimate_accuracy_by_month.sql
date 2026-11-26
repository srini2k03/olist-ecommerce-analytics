-- 23_estimate_accuracy_by_month.sql (BigQuery version)
-- Same question, method and output as sql/analytics/delivery/23_estimate_accuracy_by_month.sql.
-- Dialect notes: exact median via PERCENTILE_CONT(x, 0.5) OVER (PARTITION BY purchase_month)
--   on the filtered rows, carried through the GROUP BY with ANY_VALUE.
WITH base AS (
    SELECT
        purchase_month,
        estimated_days,
        delivery_days,
        is_late,
        PERCENTILE_CONT(delivery_days, 0.5) OVER (PARTITION BY purchase_month) AS median_days
    FROM fct_orders
    WHERE is_delivered = 1
      AND is_valid = 1
      AND purchase_date >= DATE '2017-01-01'
      AND purchase_date < DATE '2018-09-01'
)
SELECT
    purchase_month AS month,
    COUNT(*) AS delivered_orders,
    AVG(1.0 * estimated_days) AS avg_promised_days,
    AVG(1.0 * delivery_days) AS avg_actual_days,
    ANY_VALUE(median_days) AS median_actual_days,
    AVG(1.0 * (estimated_days - delivery_days)) AS avg_buffer_days,
    AVG(1.0 * is_late) AS late_rate,
    AVG(1.0 * ABS(estimated_days - delivery_days)) AS promise_mae_days
FROM base
GROUP BY purchase_month
ORDER BY purchase_month;

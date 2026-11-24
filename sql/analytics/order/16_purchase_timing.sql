-- 16_purchase_timing.sql
-- Business question: When do customers buy (day of week x hour)?
-- Method: Valid orders from fct_orders by weekday and hour of the purchase timestamp,
--   with each cell's share of all orders and each weekday's share.
-- Expected output: 7 x 24 = 168 rows; order shares sum to 1. Weekday daytime dominates.
-- Dialect notes: EXTRACT(DOW ...) is 0 = Sunday on both engines; cast to INTEGER.
WITH cells AS (
    SELECT
        CAST(EXTRACT(DOW FROM purchased_at) AS INTEGER) AS day_of_week,
        CAST(EXTRACT(HOUR FROM purchased_at) AS INTEGER) AS hour_of_day,
        COUNT(*) AS orders
    FROM fct_orders
    WHERE is_valid = 1
    GROUP BY 1, 2
)
SELECT
    day_of_week,
    CASE day_of_week
        WHEN 0 THEN 'Sun' WHEN 1 THEN 'Mon' WHEN 2 THEN 'Tue' WHEN 3 THEN 'Wed'
        WHEN 4 THEN 'Thu' WHEN 5 THEN 'Fri' ELSE 'Sat'
    END AS day_name,
    hour_of_day,
    orders,
    1.0 * orders / SUM(orders) OVER () AS order_share,
    1.0 * SUM(orders) OVER (PARTITION BY day_of_week) / SUM(orders) OVER () AS day_share
FROM cells
ORDER BY day_of_week, hour_of_day;

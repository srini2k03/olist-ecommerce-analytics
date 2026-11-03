-- 16_purchase_timing.sql
-- Business question: When do customers shop? This guides when to send campaigns and
--   when to staff customer service.
-- Method: Valid orders by day of week (0 = Sunday, the same in both engines) and hour of
--   the purchase timestamp. Share of all orders per cell, plus each day's share of
--   weekly orders through a PARTITION BY window.
-- Expected output: 168 rows (7 days x 24 hours) for a heatmap. Weekday daytime, Monday
--   and Tuesday especially, should be busiest; the small hours should be quietest.
-- Dialect notes: EXTRACT(DOW ...) uses Sunday = 0 in both DuckDB and Postgres. EXTRACT
--   returns NUMERIC in Postgres, so it is cast to INTEGER.
WITH timing AS (
    SELECT
        CAST(EXTRACT(DOW FROM order_purchase_timestamp) AS INTEGER) AS day_of_week,
        CAST(EXTRACT(HOUR FROM order_purchase_timestamp) AS INTEGER) AS hour_of_day
    FROM orders
    WHERE order_status NOT IN ('canceled', 'unavailable')
),
cells AS (
    SELECT day_of_week, hour_of_day, COUNT(*) AS orders
    FROM timing
    GROUP BY day_of_week, hour_of_day
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

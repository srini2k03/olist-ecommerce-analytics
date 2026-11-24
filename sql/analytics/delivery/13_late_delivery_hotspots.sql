-- 13_late_delivery_hotspots.sql
-- Business question: Which seller-state -> customer-state lanes produce the most late
--   deliveries, in rate and in volume?
-- Method: Delivered orders with their main seller's state (the seller of item 1, as in
--   fct_orders). Lane = seller_state -> customer_state. Per lane: orders, late orders,
--   late rate, mean days late when late, and the lane's share of ALL late orders
--   (computed before the 200-order filter). ROW_NUMBER ranks lanes by late rate.
-- Expected output: top 20 lanes with at least 200 orders.
-- Dialect notes: string concatenation with ||.
WITH lanes AS (
    SELECT
        seller_state || ' -> ' || customer_state AS lane,
        delay_days,
        is_late
    FROM fct_orders
    WHERE is_delivered = 1
      AND seller_state IS NOT NULL
),
by_lane AS (
    SELECT
        lane,
        COUNT(*) AS orders,
        SUM(is_late) AS late_orders,
        AVG(1.0 * is_late) AS late_rate,
        AVG(CASE WHEN is_late = 1 THEN 1.0 * delay_days END) AS avg_days_late_when_late
    FROM lanes
    GROUP BY lane
),
with_share AS (
    -- Share over every lane, before the 200-order filter: a window in the filtered
    -- SELECT below would only see the late orders of lanes that pass the filter.
    SELECT
        *,
        1.0 * late_orders / SUM(late_orders) OVER () AS share_of_all_late
    FROM by_lane
),
ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (ORDER BY late_rate DESC, orders DESC, lane) AS hotspot_rank
    FROM with_share
    WHERE orders >= 200
)
SELECT
    hotspot_rank,
    lane,
    orders,
    late_orders,
    late_rate,
    avg_days_late_when_late,
    share_of_all_late
FROM ranked
WHERE hotspot_rank <= 20
ORDER BY hotspot_rank;

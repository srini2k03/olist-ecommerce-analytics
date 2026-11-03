-- 13_late_delivery_hotspots.sql
-- Business question: Which seller-state to customer-state shipping lanes are late most
--   often? These are the lanes where logistics fixes or padded estimates pay off most.
-- Method: Delivered orders, each assigned to the seller of its first item. Lane =
--   seller_state -> customer_state. Lanes need at least 200 orders. We report the late
--   rate, the mean delay of late orders and the lane's share of all late orders, then
--   take the 20 worst lanes by late rate (ROW_NUMBER for a deterministic cut).
-- Expected output: 20 rows ordered by late rate. Long lanes from SP to the north-east
--   are expected near the top.
-- Dialect notes: DATE - DATE; FILTER is avoided in favour of CASE for clarity.
WITH first_seller AS (
    SELECT order_id, seller_id
    FROM order_items
    WHERE order_item_id = 1
),
lanes AS (
    SELECT
        s.seller_state || ' -> ' || c.customer_state AS lane,
        s.seller_state,
        c.customer_state,
        CAST(o.order_delivered_customer_date AS DATE)
            - CAST(o.order_estimated_delivery_date AS DATE) AS delay_days
    FROM orders AS o
    JOIN first_seller AS fs ON fs.order_id = o.order_id
    JOIN sellers AS s ON s.seller_id = fs.seller_id
    JOIN customers AS c ON c.customer_id = o.customer_id
    WHERE o.order_status = 'delivered'
      AND o.order_delivered_customer_date IS NOT NULL
      AND o.order_estimated_delivery_date IS NOT NULL
),
by_lane AS (
    SELECT
        lane,
        seller_state,
        customer_state,
        COUNT(*) AS orders,
        SUM(CASE WHEN delay_days > 0 THEN 1 ELSE 0 END) AS late_orders,
        AVG(CASE WHEN delay_days > 0 THEN 1.0 ELSE 0.0 END) AS late_rate,
        AVG(CASE WHEN delay_days > 0 THEN 1.0 * delay_days END) AS avg_days_late_when_late
    FROM lanes
    GROUP BY lane, seller_state, customer_state
),
ranked AS (
    SELECT
        *,
        1.0 * late_orders / SUM(late_orders) OVER () AS share_of_all_late,
        ROW_NUMBER() OVER (ORDER BY late_rate DESC, orders DESC, lane) AS hotspot_rank
    FROM by_lane
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

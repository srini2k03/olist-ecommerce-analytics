-- 05_seller_league.sql
-- Business question: Who are the biggest sellers, and how do they rank on review score
--   and punctuality relative to each other?
-- Method: Item revenue per seller from fct_order_items (valid orders), then per
--   (seller, order) lateness and hand-over days from fct_orders. Sellers with at least
--   100 orders form the league; RANK by revenue, PERCENT_RANK for review and on-time
--   performance. This is a revenue league; the uncertainty-aware ranking that does not
--   reward size alone is 21_seller_scorecard + olist.analytics.sellers.
-- Expected output: top 25 sellers by revenue with state, orders, revenue, mean review,
--   late rate, median hand-over days and the two percentiles.
-- Dialect notes: PERCENT_RANK and PERCENTILE_CONT exist on both engines.
WITH seller_orders AS (
    SELECT
        i.seller_id,
        i.order_id,
        SUM(i.item_revenue) AS revenue
    FROM fct_order_items AS i
    WHERE i.is_valid = 1
    GROUP BY i.seller_id, i.order_id
),
league AS (
    SELECT
        so.seller_id,
        COUNT(*) AS orders,
        SUM(so.revenue) AS revenue,
        AVG(o.review_score) AS avg_review,
        1.0 * SUM(o.is_late) / NULLIF(SUM(o.is_delivered), 0) AS late_rate,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY o.handling_days) AS median_handover_days
    FROM seller_orders AS so
    JOIN fct_orders AS o ON o.order_id = so.order_id
    GROUP BY so.seller_id
    HAVING COUNT(*) >= 100
),
ranked AS (
    SELECT
        l.*,
        s.state AS seller_state,
        RANK() OVER (ORDER BY l.revenue DESC) AS revenue_rank,
        PERCENT_RANK() OVER (ORDER BY l.avg_review) AS review_percentile,
        PERCENT_RANK() OVER (ORDER BY l.late_rate DESC) AS on_time_percentile,
        COUNT(*) OVER () AS sellers_in_league
    FROM league AS l
    JOIN dim_sellers AS s ON s.seller_id = l.seller_id
)
SELECT
    revenue_rank,
    seller_id,
    seller_state,
    orders,
    revenue,
    avg_review,
    late_rate,
    median_handover_days,
    review_percentile,
    on_time_percentile,
    sellers_in_league
FROM ranked
WHERE revenue_rank <= 25
ORDER BY revenue_rank, seller_id;

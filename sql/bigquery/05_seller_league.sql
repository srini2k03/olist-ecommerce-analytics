-- 05_seller_league.sql (BigQuery version)
-- Same question, method and output as sql/analytics/seller/05_seller_league.sql.
-- Dialect notes: exact median via PERCENTILE_CONT(x, 0.5) OVER (PARTITION BY seller_id) on
--   the joined rows, carried through the GROUP BY with ANY_VALUE.
WITH seller_orders AS (
    SELECT
        i.seller_id,
        i.order_id,
        SUM(i.item_revenue) AS revenue
    FROM fct_order_items AS i
    WHERE i.is_valid = 1
    GROUP BY i.seller_id, i.order_id
),
joined AS (
    SELECT
        so.seller_id,
        so.revenue,
        o.review_score,
        o.is_late,
        o.is_delivered,
        PERCENTILE_CONT(o.handling_days, 0.5) OVER (PARTITION BY so.seller_id) AS median_days
    FROM seller_orders AS so
    JOIN fct_orders AS o ON o.order_id = so.order_id
),
league AS (
    SELECT
        seller_id,
        COUNT(*) AS orders,
        SUM(revenue) AS revenue,
        AVG(review_score) AS avg_review,
        1.0 * SUM(is_late) / NULLIF(SUM(is_delivered), 0) AS late_rate,
        ANY_VALUE(median_days) AS median_handover_days
    FROM joined
    GROUP BY seller_id
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

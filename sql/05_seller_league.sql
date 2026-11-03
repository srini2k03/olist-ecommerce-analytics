-- 05_seller_league.sql
-- Business question: Which established sellers drive the most revenue, and do the
--   biggest sellers also keep customers happy and deliver on time?
-- Method: Sellers with at least 100 valid orders. Per seller: orders, item revenue,
--   mean review score, late-delivery rate and median hand-over-to-carrier days.
--   RANK() orders sellers by revenue; PERCENT_RANK() places each seller within the
--   league on review score and lateness so the trade-off is visible.
-- Expected output: the top 25 sellers by revenue with revenue_rank,
--   review_percentile (1 = best) and on_time_percentile (1 = best).
-- Dialect notes: PERCENTILE_CONT ... WITHIN GROUP works in both engines.
WITH order_review AS (
    SELECT order_id, AVG(review_score) AS review_score
    FROM order_reviews
    GROUP BY order_id
),
seller_orders AS (
    SELECT
        oi.seller_id,
        oi.order_id,
        SUM(oi.price) AS revenue,
        MAX(CASE
            WHEN CAST(o.order_delivered_customer_date AS DATE)
                > CAST(o.order_estimated_delivery_date AS DATE) THEN 1
            ELSE 0
        END) AS is_late,
        MAX(CASE WHEN o.order_delivered_customer_date IS NOT NULL THEN 1 ELSE 0 END)
            AS is_delivered,
        MAX(CAST(o.order_delivered_carrier_date AS DATE) - CAST(o.order_approved_at AS DATE))
            AS handover_days
    FROM order_items AS oi
    JOIN orders AS o ON o.order_id = oi.order_id
    WHERE o.order_status NOT IN ('canceled', 'unavailable')
    GROUP BY oi.seller_id, oi.order_id
),
league AS (
    SELECT
        so.seller_id,
        COUNT(*) AS orders,
        SUM(so.revenue) AS revenue,
        AVG(r.review_score) AS avg_review,
        1.0 * SUM(so.is_late) / NULLIF(SUM(so.is_delivered), 0) AS late_rate,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY so.handover_days) AS median_handover_days
    FROM seller_orders AS so
    LEFT JOIN order_review AS r ON r.order_id = so.order_id
    GROUP BY so.seller_id
    HAVING COUNT(*) >= 100
),
ranked AS (
    SELECT
        l.*,
        s.seller_state,
        RANK() OVER (ORDER BY l.revenue DESC) AS revenue_rank,
        PERCENT_RANK() OVER (ORDER BY l.avg_review) AS review_percentile,
        PERCENT_RANK() OVER (ORDER BY l.late_rate DESC) AS on_time_percentile,
        COUNT(*) OVER () AS sellers_in_league
    FROM league AS l
    JOIN sellers AS s ON s.seller_id = l.seller_id
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

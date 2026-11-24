-- 06_category_pareto.sql
-- Business question: Which product categories drive revenue, and how concentrated is it?
-- Method: Item revenue (price, no freight) per English category from fct_order_items
--   (valid orders). Running SUM over categories ordered by revenue gives the cumulative
--   share; categories whose preceding cumulative share is under 80% form the Pareto core.
-- Expected output: one row per category (about 70) with items, orders, revenue, share,
--   cumulative share and an in_pareto_core flag. Shares sum to 1.
-- Dialect notes: ROWS UNBOUNDED PRECEDING frame for the running sum.
WITH category_revenue AS (
    SELECT
        category,
        COUNT(*) AS items,
        COUNT(DISTINCT order_id) AS orders,
        SUM(item_revenue) AS revenue
    FROM fct_order_items
    WHERE is_valid = 1
    GROUP BY category
),
shares AS (
    SELECT
        category,
        items,
        orders,
        revenue,
        revenue / SUM(revenue) OVER () AS revenue_share,
        SUM(revenue) OVER (ORDER BY revenue DESC, category ROWS UNBOUNDED PRECEDING)
            / SUM(revenue) OVER () AS cumulative_share,
        ROW_NUMBER() OVER (ORDER BY revenue DESC, category) AS revenue_rank
    FROM category_revenue
)
SELECT
    revenue_rank,
    category,
    items,
    orders,
    revenue,
    revenue_share,
    cumulative_share,
    CASE WHEN cumulative_share - revenue_share < 0.8 THEN 1 ELSE 0 END AS in_pareto_core
FROM shares
ORDER BY revenue_rank;

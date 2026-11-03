-- 06_category_pareto.sql
-- Business question: How concentrated is revenue across product categories, and which
--   categories make up the first 80%?
-- Method: Item revenue (price) by English category name (untranslated or missing
--   categories are kept under their raw name or 'unknown'). A running SUM() OVER
--   (ORDER BY revenue DESC) gives the cumulative share; a category is in the Pareto
--   core if the cumulative share before it is below 80%.
-- Expected output: one row per category (about 72 rows) with revenue_rank, revenue,
--   revenue_share, cumulative_share and in_pareto_core.
-- Dialect notes: none needed beyond standard window functions.
WITH category_revenue AS (
    SELECT
        COALESCE(t.product_category_name_english, p.product_category_name, 'unknown')
            AS category,
        COUNT(*) AS items,
        COUNT(DISTINCT oi.order_id) AS orders,
        SUM(oi.price) AS revenue
    FROM order_items AS oi
    JOIN orders AS o ON o.order_id = oi.order_id
    JOIN products AS p ON p.product_id = oi.product_id
    LEFT JOIN category_translation AS t
        ON t.product_category_name = p.product_category_name
    WHERE o.order_status NOT IN ('canceled', 'unavailable')
    GROUP BY 1
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

-- 19_category_growth.sql
-- Business question: Which categories grew fastest year over year, comparing the same
--   months so seasonality cancels out?
-- Method: Item revenue and orders per category from fct_order_items (valid orders) for
--   Jan-Aug 2017 and Jan-Aug 2018. Growth = 2018 / 2017 - 1; categories need at least 200
--   orders in 2017 so growth is not computed from tiny bases. RANK by growth.
-- Expected output: about 30-40 categories with both periods' revenue, orders and growth.
-- Dialect notes: conditional aggregation with CASE inside SUM.
WITH by_period AS (
    SELECT
        category,
        SUM(CASE WHEN purchased_at >= DATE '2017-01-01' AND purchased_at < DATE '2017-09-01'
            THEN item_revenue ELSE 0 END) AS revenue_2017,
        SUM(CASE WHEN purchased_at >= DATE '2018-01-01' AND purchased_at < DATE '2018-09-01'
            THEN item_revenue ELSE 0 END) AS revenue_2018,
        COUNT(DISTINCT CASE WHEN purchased_at >= DATE '2017-01-01'
            AND purchased_at < DATE '2017-09-01' THEN order_id END) AS orders_2017,
        COUNT(DISTINCT CASE WHEN purchased_at >= DATE '2018-01-01'
            AND purchased_at < DATE '2018-09-01' THEN order_id END) AS orders_2018
    FROM fct_order_items
    WHERE is_valid = 1
    GROUP BY category
)
SELECT
    RANK() OVER (ORDER BY revenue_2018 / revenue_2017 DESC) AS growth_rank,
    category,
    orders_2017,
    orders_2018,
    revenue_2017,
    revenue_2018,
    revenue_2018 / revenue_2017 - 1 AS revenue_growth,
    1.0 * orders_2018 / orders_2017 - 1 AS order_growth
FROM by_period
WHERE orders_2017 >= 200
ORDER BY growth_rank, category;

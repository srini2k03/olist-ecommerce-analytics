-- dim_customers
-- Layer: marts
-- Grain: one row per customer_unique_id (a person), including people whose only orders
--   were cancelled (n_orders = 0, cohort_month NULL).
-- Purpose: customer attributes, acquisition cohort and lifetime totals over valid orders.
-- Depends on: stg_customers, fct_orders
-- Definitions: state/zip are those of the first valid order (a person can order from
--   several addresses); lifetime_gmv and lifetime_item_revenue sum valid orders only.
WITH valid AS (
    SELECT *
    FROM fct_orders
    WHERE is_valid = 1
),
per_customer AS (
    SELECT
        customer_unique_id,
        MIN(purchased_at) AS first_order_at,
        MAX(purchased_at) AS last_order_at,
        COUNT(*) AS n_orders,
        SUM(gmv) AS lifetime_gmv,
        SUM(item_revenue) AS lifetime_item_revenue,
        AVG(review_score) AS avg_review
    FROM valid
    GROUP BY customer_unique_id
),
first_order AS (
    SELECT customer_unique_id, customer_state, customer_zip_prefix, cohort_month
    FROM valid
    WHERE customer_order_seq = 1
),
people AS (
    SELECT customer_unique_id, MIN(state) AS any_state, MIN(zip_prefix) AS any_zip
    FROM stg_customers
    GROUP BY customer_unique_id
)
SELECT
    p.customer_unique_id,
    COALESCE(f.customer_state, p.any_state) AS customer_state,
    COALESCE(f.customer_zip_prefix, p.any_zip) AS zip_prefix,
    f.cohort_month,
    pc.first_order_at,
    pc.last_order_at,
    COALESCE(pc.n_orders, 0) AS n_orders,
    COALESCE(pc.lifetime_gmv, 0) AS lifetime_gmv,
    COALESCE(pc.lifetime_item_revenue, 0) AS lifetime_item_revenue,
    pc.avg_review,
    CASE WHEN pc.n_orders > 1 THEN 1 ELSE 0 END AS is_repeat_customer
FROM people AS p
LEFT JOIN per_customer AS pc ON pc.customer_unique_id = p.customer_unique_id
LEFT JOIN first_order AS f ON f.customer_unique_id = p.customer_unique_id

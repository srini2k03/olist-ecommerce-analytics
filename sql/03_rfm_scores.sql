-- 03_rfm_scores.sql
-- Business question: How is the customer base split across RFM segments, and how much
--   revenue does each segment hold?
-- Method: Per customer_unique_id: recency = days from last order to the reference date
--   2018-09-01, frequency = number of valid orders, monetary = total price + freight.
--   R and M are scored 1-5 with NTILE(5). F is NOT scored with NTILE: about 97% of
--   customers have exactly one order, so NTILE would hand different scores to identical
--   customers. F is the order count capped at 5 instead. Scores map to named segments
--   with a CASE rule, and windowed SUM() OVER () gives each segment's share.
-- Expected output: one row per segment (about 7 rows): segment, customers,
--   customer_share, revenue_share, avg_recency_days, avg_frequency, avg_monetary.
-- Dialect notes: DATE - DATE returns whole days in both engines.
WITH order_value AS (
    SELECT order_id, SUM(price + freight_value) AS order_value
    FROM order_items
    GROUP BY order_id
),
customer_rfm AS (
    SELECT
        c.customer_unique_id,
        DATE '2018-09-01' - CAST(MAX(o.order_purchase_timestamp) AS DATE) AS recency_days,
        COUNT(DISTINCT o.order_id) AS frequency,
        SUM(ov.order_value) AS monetary
    FROM orders AS o
    JOIN customers AS c ON c.customer_id = o.customer_id
    JOIN order_value AS ov ON ov.order_id = o.order_id
    WHERE o.order_status NOT IN ('canceled', 'unavailable')
      AND o.order_purchase_timestamp < DATE '2018-09-01'
    GROUP BY c.customer_unique_id
),
scored AS (
    SELECT
        customer_unique_id,
        recency_days,
        frequency,
        monetary,
        NTILE(5) OVER (ORDER BY recency_days DESC, customer_unique_id) AS r_score,
        LEAST(frequency, 5) AS f_score,
        NTILE(5) OVER (ORDER BY monetary, customer_unique_id) AS m_score
    FROM customer_rfm
),
segmented AS (
    SELECT
        *,
        CASE
            WHEN f_score >= 2 AND r_score >= 4 THEN 'Champions'
            WHEN f_score >= 2 THEN 'Loyal, lapsing'
            WHEN r_score >= 4 AND m_score >= 4 THEN 'New big spenders'
            WHEN r_score >= 4 THEN 'New low spenders'
            WHEN r_score <= 2 AND m_score >= 4 THEN 'At-risk big spenders'
            WHEN r_score <= 2 THEN 'Lost one-timers'
            ELSE 'Fading one-timers'
        END AS segment
    FROM scored
)
SELECT
    segment,
    COUNT(*) AS customers,
    1.0 * COUNT(*) / SUM(COUNT(*)) OVER () AS customer_share,
    SUM(monetary) / SUM(SUM(monetary)) OVER () AS revenue_share,
    AVG(recency_days) AS avg_recency_days,
    AVG(frequency) AS avg_frequency,
    AVG(monetary) AS avg_monetary
FROM segmented
GROUP BY segment
ORDER BY revenue_share DESC, segment;

-- 03_rfm_scores.sql
-- Business question: How are customers distributed across classic RFM segments, and how
--   much revenue does each segment hold?
-- Method: Valid orders before 2018-09-01. Recency = days from the last order to
--   2018-09-01, frequency = number of orders, monetary = GMV. Recency and monetary are
--   NTILE(5) scored; frequency is capped at 5 (97% of customers have a single order, so a
--   quintile of frequency would be meaningless). Rule-based segment names on the scores.
-- Expected output: one row per segment with customers, customer_share, revenue_share and
--   average R, F, M. Shares sum to 1.
-- Dialect notes: the NTILE ORDER BY includes customer_unique_id so ties split
--   deterministically on both engines.
WITH customer_rfm AS (
    SELECT
        customer_unique_id,
        DATE '2018-09-01' - MAX(purchase_date) AS recency_days,
        COUNT(*) AS frequency,
        SUM(gmv) AS monetary
    FROM fct_orders
    WHERE is_valid = 1
      AND purchase_date < DATE '2018-09-01'
    GROUP BY customer_unique_id
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
    AVG(1.0 * recency_days) AS avg_recency_days,
    AVG(1.0 * frequency) AS avg_frequency,
    AVG(monetary) AS avg_monetary
FROM segmented
GROUP BY segment
ORDER BY revenue_share DESC, segment;

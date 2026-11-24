-- 14_review_sentiment_proxies.sql
-- Business question: What do unhappy customers write about: delivery or the product?
-- Method: Every review in fct_reviews. Lower-cased comments are flagged with Portuguese
--   keyword lists for delivery problems, product problems and praise; rates are per
--   commented review, by star rating.
-- Expected output: 5 rows (1 to 5 stars) with comment share, mean comment length and the
--   three mention rates. Delivery mentions should dominate low scores.
-- Dialect notes: LOWER(...) LIKE instead of ILIKE or regex, which differ across engines.
WITH flagged AS (
    SELECT
        review_score,
        has_comment,
        LENGTH(LOWER(comment_message)) AS comment_length,
        CASE
            WHEN LOWER(comment_message) LIKE '%não recebi%'
                OR LOWER(comment_message) LIKE '%nao recebi%'
                OR LOWER(comment_message) LIKE '%atras%'
                OR LOWER(comment_message) LIKE '%ainda não%'
                OR LOWER(comment_message) LIKE '%entreg%' THEN 1.0
            ELSE 0.0
        END AS mentions_delivery,
        CASE
            WHEN LOWER(comment_message) LIKE '%defeito%'
                OR LOWER(comment_message) LIKE '%quebrad%'
                OR LOWER(comment_message) LIKE '%errad%'
                OR LOWER(comment_message) LIKE '%falsificad%' THEN 1.0
            ELSE 0.0
        END AS mentions_product_problem,
        CASE
            WHEN LOWER(comment_message) LIKE '%recomendo%'
                OR LOWER(comment_message) LIKE '%ótimo%'
                OR LOWER(comment_message) LIKE '%otimo%'
                OR LOWER(comment_message) LIKE '%excelente%'
                OR LOWER(comment_message) LIKE '%perfeito%'
                OR LOWER(comment_message) LIKE '%rápid%'
                OR LOWER(comment_message) LIKE '%rapid%' THEN 1.0
            ELSE 0.0
        END AS mentions_praise
    FROM fct_reviews
)
SELECT
    review_score,
    COUNT(*) AS reviews,
    AVG(1.0 * has_comment) AS share_with_comment,
    AVG(1.0 * comment_length) AS avg_comment_length,
    SUM(mentions_delivery) / NULLIF(SUM(has_comment), 0) AS delivery_mention_rate,
    SUM(mentions_product_problem) / NULLIF(SUM(has_comment), 0) AS product_problem_rate,
    SUM(mentions_praise) / NULLIF(SUM(has_comment), 0) AS praise_rate
FROM flagged
GROUP BY review_score
ORDER BY review_score;

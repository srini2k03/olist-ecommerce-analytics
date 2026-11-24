-- 14_review_sentiment_proxies.sql
-- Business question: What do unhappy customers write about: delivery or the product?
-- Method: Every review in fct_reviews. Lower-cased comments are flagged with Portuguese
--   keyword lists for delivery problems, product problems and praise; rates are per
--   commented review, by star rating.
-- Expected output: 5 rows (1 to 5 stars) with comment share, mean comment length and the
--   three mention rates. Delivery mentions should dominate low scores.
-- Dialect notes: LOWER(...) LIKE instead of ILIKE or regex, which differ across engines.
WITH reviews AS (
    -- Lower-case each comment once; repeating LOWER() in every LIKE made this the slowest
    -- query in the benchmark.
    SELECT review_score, has_comment, LOWER(comment_message) AS msg
    FROM fct_reviews
),
flagged AS (
    SELECT
        review_score,
        has_comment,
        LENGTH(msg) AS comment_length,
        CASE
            WHEN msg LIKE '%não recebi%' OR msg LIKE '%nao recebi%' OR msg LIKE '%atras%'
                OR msg LIKE '%ainda não%' OR msg LIKE '%entreg%' THEN 1.0
            ELSE 0.0
        END AS mentions_delivery,
        CASE
            WHEN msg LIKE '%defeito%' OR msg LIKE '%quebrad%' OR msg LIKE '%errad%'
                OR msg LIKE '%falsificad%' THEN 1.0
            ELSE 0.0
        END AS mentions_product_problem,
        CASE
            WHEN msg LIKE '%recomendo%' OR msg LIKE '%ótimo%' OR msg LIKE '%otimo%'
                OR msg LIKE '%excelente%' OR msg LIKE '%perfeito%' OR msg LIKE '%rápid%'
                OR msg LIKE '%rapid%' THEN 1.0
            ELSE 0.0
        END AS mentions_praise
    FROM reviews
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

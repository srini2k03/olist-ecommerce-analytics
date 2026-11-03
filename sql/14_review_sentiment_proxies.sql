-- 14_review_sentiment_proxies.sql
-- Business question: Without a sentiment model, what do review texts tell us about why
--   customers are unhappy? Do complaints about delivery dominate low scores?
-- Method: Simple Portuguese keyword proxies on the lower-cased comment: delivery
--   complaints ('não recebi', 'atras', 'ainda não', 'entreg'), product complaints
--   ('defeito', 'quebrad', 'errad', 'falsificad'), and praise ('recomendo', 'ótimo',
--   'excelente', 'perfeito', 'rápid'). For each review score we report how many reviews
--   carry a comment, the mean comment length, and the share mentioning each theme
--   (as a share of reviews that have a comment).
-- Expected output: 5 rows (score 1-5). Delivery terms should be far more common in 1-2
--   star comments, and praise terms in 4-5 star comments.
-- Dialect notes: LOWER + LIKE is used instead of ILIKE/regex so both engines agree.
WITH reviews AS (
    SELECT
        review_score,
        LOWER(review_comment_message) AS msg
    FROM order_reviews
),
flagged AS (
    SELECT
        review_score,
        CASE WHEN msg IS NOT NULL THEN 1.0 ELSE 0.0 END AS has_comment,
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
    AVG(has_comment) AS share_with_comment,
    AVG(1.0 * comment_length) AS avg_comment_length,
    SUM(mentions_delivery) / NULLIF(SUM(has_comment), 0) AS delivery_mention_rate,
    SUM(mentions_product_problem) / NULLIF(SUM(has_comment), 0) AS product_problem_rate,
    SUM(mentions_praise) / NULLIF(SUM(has_comment), 0) AS praise_rate
FROM flagged
GROUP BY review_score
ORDER BY review_score;

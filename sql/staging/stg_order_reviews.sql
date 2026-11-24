-- stg_order_reviews
-- Layer: staging
-- Grain: one row per (review_id, order_id). The raw review_id is NOT unique: 789 review ids
--   are attached to more than one order (one survey answered for several orders), and 547
--   orders have more than one review.
-- Purpose: review scores with the timestamps that matter for leakage-safe features.
-- Depends on: order_reviews
-- Transformations: exact duplicate (review_id, order_id) rows are dropped, keeping the
--   latest answer; has_comment marks a non-empty comment.
WITH ranked AS (
    SELECT
        review_id,
        order_id,
        review_score,
        review_comment_message,
        review_creation_date,
        review_answer_timestamp,
        ROW_NUMBER() OVER (
            PARTITION BY review_id, order_id
            ORDER BY review_answer_timestamp DESC
        ) AS rn
    FROM order_reviews
)
SELECT
    review_id,
    order_id,
    review_score,
    CASE WHEN review_comment_message IS NOT NULL THEN 1 ELSE 0 END AS has_comment,
    review_comment_message AS comment_message,
    review_creation_date AS survey_sent_at,
    review_answer_timestamp AS answered_at
FROM ranked
WHERE rn = 1

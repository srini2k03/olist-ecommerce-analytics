-- fct_reviews
-- Layer: marts
-- Grain: one row per (review_id, order_id).
-- Purpose: review text and score with the order's delivery outcome, for review analytics.
-- Depends on: stg_order_reviews, fct_orders
SELECT
    r.review_id,
    r.order_id,
    r.review_score,
    r.has_comment,
    r.comment_message,
    r.survey_sent_at,
    r.answered_at,
    o.is_valid,
    o.is_delivered,
    o.is_late,
    o.delay_days,
    o.category,
    o.customer_state
FROM stg_order_reviews AS r
JOIN fct_orders AS o ON o.order_id = r.order_id

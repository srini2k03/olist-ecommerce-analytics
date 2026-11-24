-- stg_orders
-- Layer: staging
-- Grain: one row per order_id.
-- Purpose: order header with normalised dates, status flags and data-quality flags.
-- Depends on: orders
-- Transformations: status lower-cased; purchase date and month derived once here so every
--   downstream model buckets time the same way; is_cancelled covers both 'canceled' and
--   'unavailable' (neither became a sale); is_delivered needs status 'delivered' AND a
--   delivery timestamp (8 'delivered' orders have none). dq_* flags keep known timestamp
--   anomalies visible instead of silently dropping the rows.
SELECT
    order_id,
    customer_id,
    LOWER(TRIM(order_status)) AS order_status,
    order_purchase_timestamp AS purchased_at,
    order_approved_at AS approved_at,
    order_delivered_carrier_date AS shipped_at,
    order_delivered_customer_date AS delivered_at,
    order_estimated_delivery_date AS estimated_delivery_at,
    CAST(order_purchase_timestamp AS DATE) AS purchase_date,
    CAST(date_trunc('month', order_purchase_timestamp) AS DATE) AS purchase_month,
    CASE WHEN LOWER(TRIM(order_status)) IN ('canceled', 'unavailable') THEN 1 ELSE 0 END
        AS is_cancelled,
    CASE
        WHEN LOWER(TRIM(order_status)) = 'delivered'
            AND order_delivered_customer_date IS NOT NULL THEN 1
        ELSE 0
    END AS is_delivered,
    CASE WHEN order_delivered_carrier_date < order_approved_at THEN 1 ELSE 0 END
        AS dq_shipped_before_approved,
    CASE WHEN order_delivered_customer_date < order_delivered_carrier_date THEN 1 ELSE 0 END
        AS dq_delivered_before_shipped,
    CASE
        WHEN LOWER(TRIM(order_status)) = 'delivered'
            AND order_delivered_customer_date IS NULL THEN 1
        ELSE 0
    END AS dq_delivered_without_date
FROM orders

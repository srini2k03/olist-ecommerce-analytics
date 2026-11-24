-- fct_orders
-- Layer: marts
-- Grain: one row per order_id (every order, including cancelled ones; filter on is_valid).
-- Purpose: the single source of truth for order-level money, delivery, review and customer
--   sequence. Analytics queries and the metric layer read this table instead of re-deriving
--   order value from the raw tables.
-- Depends on: stg_orders, stg_order_items, stg_order_payments, stg_order_reviews,
--   stg_customers, stg_products, stg_sellers, stg_geolocation
-- Money columns (never mix them):
--   item_revenue  = SUM(price) of the order's items (merchandise value, paid to sellers)
--   freight_value = SUM(freight_value) charged to the customer for shipping
--   gmv           = item_revenue + freight_value (gross merchandise value incl. freight)
--   payment_value = SUM(payment_value) actually charged (includes instalment interest and
--                   vouchers; differs from gmv on a few hundred orders)
-- Definitions: is_valid = not cancelled/unavailable AND has at least one item. Delivery
--   day counts are whole calendar days (DATE - DATE). is_late = delivered after the
--   estimated date (NULL when not delivered). customer_order_seq and cohort_month count
--   valid orders only.
WITH items AS (
    SELECT
        order_id,
        COUNT(*) AS n_items,
        COUNT(DISTINCT product_id) AS n_products,
        COUNT(DISTINCT seller_id) AS n_sellers,
        SUM(price) AS item_revenue,
        SUM(freight_value) AS freight_value
    FROM stg_order_items
    GROUP BY order_id
),
first_item AS (
    SELECT order_id, product_id, seller_id
    FROM (
        SELECT
            order_id,
            product_id,
            seller_id,
            ROW_NUMBER() OVER (PARTITION BY order_id ORDER BY order_item_id) AS rn
        FROM stg_order_items
    ) AS t
    WHERE rn = 1
),
payments AS (
    SELECT
        order_id,
        SUM(payment_value) AS payment_value,
        COUNT(*) AS n_payments,
        MAX(installments) AS installments,
        MAX(CASE WHEN payment_type = 'voucher' THEN 1 ELSE 0 END) AS used_voucher
    FROM stg_order_payments
    GROUP BY order_id
),
main_payment AS (
    SELECT order_id, payment_type AS main_payment_type
    FROM (
        SELECT
            order_id,
            payment_type,
            ROW_NUMBER() OVER (
                PARTITION BY order_id ORDER BY payment_value DESC, payment_sequential
            ) AS rn
        FROM stg_order_payments
    ) AS t
    WHERE rn = 1
),
reviews AS (
    SELECT
        order_id,
        AVG(1.0 * review_score) AS review_score,
        COUNT(*) AS n_reviews,
        MIN(survey_sent_at) AS review_sent_at,
        MAX(answered_at) AS review_answered_at,
        MAX(has_comment) AS has_comment
    FROM stg_order_reviews
    GROUP BY order_id
),
base AS (
    SELECT
        o.order_id,
        o.customer_id,
        c.customer_unique_id,
        c.state AS customer_state,
        c.zip_prefix AS customer_zip_prefix,
        o.order_status,
        o.is_cancelled,
        CASE WHEN o.is_cancelled = 0 AND i.n_items IS NOT NULL THEN 1 ELSE 0 END AS is_valid,
        o.is_delivered,
        o.purchased_at,
        o.approved_at,
        o.shipped_at,
        o.delivered_at,
        o.estimated_delivery_at,
        o.purchase_date,
        o.purchase_month,
        i.n_items,
        i.n_products,
        i.n_sellers,
        i.item_revenue,
        i.freight_value,
        i.item_revenue + i.freight_value AS gmv,
        p.payment_value,
        p.n_payments,
        p.installments,
        mp.main_payment_type,
        p.used_voucher,
        fi.product_id AS main_product_id,
        fi.seller_id AS main_seller_id,
        pr.category,
        pr.weight_g AS main_product_weight_g,
        pr.volume_cm3 AS main_product_volume_cm3,
        s.state AS seller_state,
        r.review_score,
        r.n_reviews,
        r.review_sent_at,
        r.review_answered_at,
        r.has_comment,
        cg.lat AS customer_lat,
        cg.lng AS customer_lng,
        sg.lat AS seller_lat,
        sg.lng AS seller_lng,
        o.dq_shipped_before_approved,
        o.dq_delivered_before_shipped
    FROM stg_orders AS o
    JOIN stg_customers AS c ON c.customer_id = o.customer_id
    LEFT JOIN items AS i ON i.order_id = o.order_id
    LEFT JOIN first_item AS fi ON fi.order_id = o.order_id
    LEFT JOIN stg_products AS pr ON pr.product_id = fi.product_id
    LEFT JOIN stg_sellers AS s ON s.seller_id = fi.seller_id
    LEFT JOIN payments AS p ON p.order_id = o.order_id
    LEFT JOIN main_payment AS mp ON mp.order_id = o.order_id
    LEFT JOIN reviews AS r ON r.order_id = o.order_id
    LEFT JOIN stg_geolocation AS cg ON cg.zip_prefix = c.zip_prefix
    LEFT JOIN stg_geolocation AS sg ON sg.zip_prefix = s.zip_prefix
),
customer_sequence AS (
    SELECT
        order_id,
        ROW_NUMBER() OVER (
            PARTITION BY customer_unique_id ORDER BY purchased_at, order_id
        ) AS customer_order_seq,
        MIN(purchase_month) OVER (PARTITION BY customer_unique_id) AS cohort_month,
        MIN(purchased_at) OVER (PARTITION BY customer_unique_id) AS customer_first_purchase_at
    FROM base
    WHERE is_valid = 1
)
SELECT
    b.*,
    b.payment_value - b.gmv AS payment_minus_gmv,
    CASE
        WHEN b.is_delivered = 1
            THEN CAST(b.delivered_at AS DATE) - b.purchase_date
    END AS delivery_days,
    CAST(b.estimated_delivery_at AS DATE) - b.purchase_date AS estimated_days,
    CASE
        WHEN b.is_delivered = 1
            THEN CAST(b.delivered_at AS DATE) - CAST(b.estimated_delivery_at AS DATE)
    END AS delay_days,
    CASE
        WHEN b.is_delivered = 1 AND CAST(b.delivered_at AS DATE)
            > CAST(b.estimated_delivery_at AS DATE) THEN 1
        WHEN b.is_delivered = 1 THEN 0
    END AS is_late,
    CAST(b.shipped_at AS DATE) - CAST(b.approved_at AS DATE) AS handling_days,
    CASE
        WHEN b.is_delivered = 1
            THEN CAST(b.delivered_at AS DATE) - CAST(b.shipped_at AS DATE)
    END AS carrier_days,
    2 * 6371.0 * ASIN(SQRT(
        POWER(SIN(RADIANS(b.seller_lat - b.customer_lat) / 2), 2)
        + COS(RADIANS(b.customer_lat)) * COS(RADIANS(b.seller_lat))
        * POWER(SIN(RADIANS(b.seller_lng - b.customer_lng) / 2), 2)
    )) AS distance_km,
    CASE WHEN b.customer_state = b.seller_state THEN 1 ELSE 0 END AS same_state,
    cs.customer_order_seq,
    cs.cohort_month,
    cs.customer_first_purchase_at,
    CASE WHEN cs.customer_order_seq > 1 THEN 1 ELSE 0 END AS is_repeat_order
FROM base AS b
LEFT JOIN customer_sequence AS cs ON cs.order_id = b.order_id

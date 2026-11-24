-- 26_installments_by_order_value.sql
-- Business question: Do customers lean on instalments and vouchers more for expensive
--   orders? (Credit risk and financing cost grow with the instalment count.)
-- Method: Valid orders from fct_orders in GMV bands. Per band: orders, mean GMV, share paid
--   mainly by credit card, mean maximum instalments, share with more than one instalment
--   and share using a voucher.
-- Expected output: 5 bands from 'under R$ 50' to 'R$ 500+'; instalments rise with value.
-- Dialect notes: band order is a numeric sort key.
WITH banded AS (
    SELECT
        CASE
            WHEN gmv < 50 THEN 1
            WHEN gmv < 100 THEN 2
            WHEN gmv < 200 THEN 3
            WHEN gmv < 500 THEN 4
            ELSE 5
        END AS band_order,
        gmv,
        main_payment_type,
        installments,
        used_voucher
    FROM fct_orders
    WHERE is_valid = 1
      AND payment_value IS NOT NULL
)
SELECT
    band_order,
    CASE band_order
        WHEN 1 THEN 'under R$ 50'
        WHEN 2 THEN 'R$ 50-100'
        WHEN 3 THEN 'R$ 100-200'
        WHEN 4 THEN 'R$ 200-500'
        ELSE 'R$ 500+'
    END AS gmv_band,
    COUNT(*) AS orders,
    AVG(gmv) AS avg_gmv,
    AVG(CASE WHEN main_payment_type = 'credit_card' THEN 1.0 ELSE 0.0 END) AS share_credit_card,
    AVG(1.0 * installments) AS avg_installments,
    AVG(CASE WHEN installments > 1 THEN 1.0 ELSE 0.0 END) AS share_multi_installment,
    AVG(1.0 * used_voucher) AS share_voucher
FROM banded
GROUP BY band_order
ORDER BY band_order;

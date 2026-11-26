# BigQuery parity

- Generated 2026-11-26T12:15:55+00:00 at commit `ecdbf73f2a`, BigQuery sandbox (no billing).
- **26 / 26 analytics queries return the same result on BigQuery as on DuckDB** (cell by cell, relative tolerance 1e-6).
- 7 marts loaded from Parquet: fct_orders (99,441 rows), fct_order_items (112,650 rows), fct_payments (103,886 rows), fct_reviews (99,224 rows), dim_customers (96,096 rows), dim_products (32,951 rows), dim_sellers (3,095 rows).
- Total bytes processed by the queries: 169.9 MB (the sandbox allows 1 TB a month).

| Query | BigQuery SQL | Rows | Match | DuckDB ms | BigQuery ms | MB processed |
|---|---|---|---|---|---|---|
| 01_monthly_revenue_mom | translated | 20 | yes | 13 | 761 | 4.8 |
| 02_cohort_retention | translated | 207 | yes | 19 | 775 | 6.6 |
| 03_rfm_scores | translated | 7 | yes | 60 | 1173 | 5.8 |
| 04_delivery_delay_vs_review | translated | 7 | yes | 6 | 803 | 2.4 |
| 05_seller_league | hand-written | 25 | yes | 39 | 1007 | 16.1 |
| 06_category_pareto | translated | 74 | yes | 22 | 1259 | 7.3 |
| 07_payment_mix | translated | 4 | yes | 25 | 1061 | 7.3 |
| 08_freight_ratio | hand-written | 22 | yes | 8 | 742 | 6.2 |
| 09_state_delivery_sla | hand-written | 21 | yes | 9 | 817 | 4.3 |
| 10_repeat_purchase_rate | translated | 8 | yes | 22 | 903 | 6.6 |
| 11_basket_size | translated | 4 | yes | 9 | 651 | 4.7 |
| 12_time_to_second_order | hand-written | 7 | yes | 26 | 847 | 9.1 |
| 13_late_delivery_hotspots | translated | 20 | yes | 10 | 711 | 3.1 |
| 14_review_sentiment_proxies | translated | 5 | yes | 56 | 801 | 4.5 |
| 15_customer_value_deciles | translated | 10 | yes | 28 | 700 | 4.8 |
| 16_purchase_timing | translated | 168 | yes | 6 | 822 | 1.6 |
| 17_order_status_funnel | translated | 20 | yes | 5 | 734 | 3.2 |
| 18_money_reconciliation | translated | 20 | yes | 6 | 828 | 6.3 |
| 19_category_growth | translated | 22 | yes | 23 | 863 | 8.2 |
| 20_cohort_revenue | translated | 20 | yes | 21 | 917 | 7.3 |
| 21_seller_scorecard | translated | 3095 | yes | 35 | 1181 | 14.4 |
| 22_delivery_by_category | hand-written | 26 | yes | 10 | 743 | 7.8 |
| 23_estimate_accuracy_by_month | hand-written | 20 | yes | 11 | 708 | 5.5 |
| 24_category_scorecard | translated | 74 | yes | 42 | 924 | 13.8 |
| 25_review_by_delivery_status | translated | 3 | yes | 12 | 696 | 3.2 |
| 26_installments_by_order_value | translated | 5 | yes | 7 | 792 | 5.2 |

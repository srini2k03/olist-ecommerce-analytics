# BigQuery parity

- Generated 2026-11-26T12:10:46+00:00 at commit `1e87169346`, BigQuery sandbox (no billing).
- **24 / 26 analytics queries return the same result on BigQuery as on DuckDB** (cell by cell, relative tolerance 1e-6).
- 7 marts loaded from Parquet: fct_orders (99,441 rows), fct_order_items (112,650 rows), fct_payments (103,886 rows), fct_reviews (99,224 rows), dim_customers (96,096 rows), dim_products (32,951 rows), dim_sellers (3,095 rows).
- Total bytes processed by the queries: 168.8 MB (the sandbox allows 1 TB a month).

| Query | BigQuery SQL | Rows | Match | DuckDB ms | BigQuery ms | MB processed |
|---|---|---|---|---|---|---|
| 01_monthly_revenue_mom | translated | 20 | yes | 14 | 1441 | 4.8 |
| 02_cohort_retention | translated | 207 | yes | 24 | 1287 | 6.6 |
| 03_rfm_scores | translated | 7 | NO: BigQuery error: 400 No matching signature for operator *
  Argument types: FLOAT | 65 | 1287 | 6.6 |
| 04_delivery_delay_vs_review | translated | 7 | yes | 8 | 1126 | 2.4 |
| 05_seller_league | hand-written | 25 | yes | 49 | 1296 | 16.1 |
| 06_category_pareto | translated | 74 | yes | 31 | 1735 | 7.3 |
| 07_payment_mix | translated | 4 | yes | 34 | 1063 | 7.3 |
| 08_freight_ratio | hand-written | 22 | yes | 12 | 1047 | 6.2 |
| 09_state_delivery_sla | hand-written | 21 | yes | 12 | 1086 | 4.3 |
| 10_repeat_purchase_rate | translated | 8 | yes | 27 | 1006 | 6.6 |
| 11_basket_size | translated | 4 | yes | 12 | 939 | 4.7 |
| 12_time_to_second_order | hand-written | 7 | yes | 34 | 1278 | 9.1 |
| 13_late_delivery_hotspots | translated | 20 | yes | 12 | 921 | 3.1 |
| 14_review_sentiment_proxies | translated | 5 | yes | 65 | 1434 | 4.5 |
| 15_customer_value_deciles | translated | 10 | yes | 31 | 976 | 4.8 |
| 16_purchase_timing | translated | 168 | yes | 7 | 893 | 1.6 |
| 17_order_status_funnel | translated | 20 | yes | 7 | 829 | 3.2 |
| 18_money_reconciliation | translated | 20 | yes | 8 | 886 | 6.3 |
| 19_category_growth | translated | 22 | NO: BigQuery error: 400 No matching signature for operator >= for argument types: TI | 36 | 886 | 6.3 |
| 20_cohort_revenue | translated | 20 | yes | 26 | 1025 | 7.3 |
| 21_seller_scorecard | translated | 3095 | yes | 51 | 1176 | 14.4 |
| 22_delivery_by_category | hand-written | 26 | yes | 14 | 922 | 7.8 |
| 23_estimate_accuracy_by_month | hand-written | 20 | yes | 14 | 1168 | 5.5 |
| 24_category_scorecard | translated | 74 | yes | 57 | 1394 | 13.8 |
| 25_review_by_delivery_status | translated | 3 | yes | 10 | 1046 | 3.2 |
| 26_installments_by_order_value | translated | 5 | yes | 9 | 930 | 5.2 |

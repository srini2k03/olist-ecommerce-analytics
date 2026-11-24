# Data quality

The pipeline runs 75 checks (45 on the raw tables, 9 on staging, 21 contract checks on the
marts) and stops on any error-severity failure. The full, generated report is
[`results/data_quality/report.md`](../results/data_quality/report.md); the dashboard shows
its status in the sidebar and on the overview page. Current status: 64 pass, 11 warn (known
source issues handled in staging), 0 fail.

| Issue found in the source data | Size | Handling |
|---|---|---|
| Customer and seller zip prefixes lost their leading zero in the CSVs | 23,995 customers, 1,027 sellers | left-padded to 5 digits in staging; customers without coordinates fell from 24,265 to 280 |
| Geolocation has many points per prefix, some outside Brazil | 1,000,163 rows, 19,015 prefixes | points outside Brazil dropped, median point per prefix |
| `review_id` is shared across orders; some orders have several reviews | 814 extra rows; 547 orders | grain is (review_id, order_id); orders use the mean score |
| Orders without items (mostly unavailable/canceled) | 775 | not sales: `is_valid = 0` |
| Delivered orders without a delivery date | 8 | excluded from delivery metrics |
| Carrier hand-over before approval / delivery before hand-over | 1,359 / 23 | flagged (`dq_*` columns), kept |
| Categories missing from the translation table | 2 categories, 13 products | translated in staging |
| Payments differing from items + freight by more than R$ 1 | 312 valid orders | reported (instalment interest, vouchers); payment value is a separate metric |
| Payments with 0 instalments | 2 | set to 1 |

Contract checks on the marts include: one row per order, item and person; item revenue,
freight and payment totals reconcile with the raw tables to the cent; GMV = item revenue +
freight on every order; every valid order has a cohort and a sequence number; one first order
per buyer; lateness only for delivered orders.

Failure behaviour is tested (`tests/test_pipeline.py`): an orphan item, a duplicated order id
or a 7-star review stop the pipeline; a missing column raises a schema error; an unparseable
timestamp is counted in the report.

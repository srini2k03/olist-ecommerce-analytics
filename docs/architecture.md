# Architecture

## Pipeline

```mermaid
flowchart LR
    subgraph sources[Raw]
        csv[9 Olist CSVs<br/>checksum-pinned mirror]
    end
    subgraph wh[DuckDB warehouse  data/olist.duckdb]
        raw[raw tables<br/>typed load]
        chk1{{45 raw checks<br/>keys, FKs, ranges,<br/>dates, completeness}}
        stg[staging<br/>stg_* x 8]
        chk2{{9 staging checks}}
        marts[marts<br/>fct_orders, fct_order_items,<br/>fct_payments, fct_reviews,<br/>dim_customers, dim_sellers,<br/>dim_products]
        chk3{{21 contract checks<br/>reconciliation, identities}}
    end
    csv -->|fetch_data.py| raw --> chk1 --> stg --> chk2 --> marts --> chk3
    chk3 --> sql[26 analytics queries<br/>sql/analytics/domain/]
    chk3 --> metrics[metric layer<br/>metrics/metrics.yml]
    chk3 --> analysis[analytics<br/>cohorts, RFM, CLV, sellers,<br/>delivery model, A/B, scenarios]
    sql --> results[(results/sql<br/>results/analytics<br/>results/data_quality)]
    metrics --> results
    analysis --> results
    marts --> bundle[app/data bundle<br/>orders + items parquet, 7 MB]
    results --> bundle
    bundle --> dash[10-page dashboard<br/>Streamlit, stlite on Pages]
    results --> docs[README, summary,<br/>docs, Pages site]
    raw -. same SQL .-> pg[(Postgres<br/>CI parity check)]
```

`python -m olist.pipeline` builds the warehouse layers and stops on any failing
error-severity check. `scripts/run_sql.py`, `scripts/run_analysis.py` and
`scripts/build_app_data.py` produce the results and the dashboard bundle; the `build-data`
workflow reruns all of them on the real data and commits the outputs.

## Layers

| Layer | What it holds | Rules |
|---|---|---|
| raw | the 9 CSVs with declared types (`olist.schema`) | missing columns or text in numeric columns raise `SchemaError`; unparseable timestamps are counted |
| staging (`stg_*`) | one model per source table | normalise types, case, zip prefixes, categories; deduplicate; flag known timestamp anomalies; no business logic |
| marts (`fct_*`, `dim_*`) | star schema for analytics | the only place that defines a valid order, money columns, delivery outcomes and customer sequence |
| analytics | `sql/analytics/<domain>/` | read marts only (tested) |

Models are plain `SELECT` statements with a header (`Layer`, `Grain`, `Purpose`,
`Depends on`, `Transformations`). The builder sorts them topologically and materialises each
with `CREATE TABLE ... AS`, which DuckDB and Postgres share. On Postgres each model is
`ANALYZE`d after creation (without statistics the planner chose nested loops and `fct_orders`
took 257 s instead of 4 s; see the benchmark).

## Why DuckDB, and why Postgres stays

DuckDB is the engine: an embedded file, no server, fast enough that the whole layered build
takes about 7 s on a laptop-class machine. Postgres is kept only as a **CI portability
check**: the same staging, marts and analytics SQL run on both engines on the real data and
must give identical results. It costs one CI job and no runtime code, and it guarantees the
SQL can move to a server warehouse; it already caught two engine-specific performance traps
(correlated `NOT EXISTS` inside `CASE`, missing planner statistics) and one ordering
nondeterminism. There is no second production database to maintain.

## Entity-relationship diagram (marts)

```mermaid
erDiagram
    dim_customers ||--o{ fct_orders : "customer_unique_id"
    fct_orders ||--|{ fct_order_items : "order_id"
    fct_orders ||--o{ fct_payments : "order_id"
    fct_orders ||--o{ fct_reviews : "order_id"
    dim_products ||--o{ fct_order_items : "product_id"
    dim_sellers ||--o{ fct_order_items : "seller_id"

    dim_customers {
        text customer_unique_id PK
        text customer_state
        date cohort_month
        int n_orders
        double lifetime_gmv
    }
    fct_orders {
        text order_id PK
        text customer_unique_id FK
        int is_valid
        double item_revenue
        double freight_value
        double gmv
        double payment_value
        int delivery_days
        int is_late
        int customer_order_seq
    }
    fct_order_items {
        text order_id FK
        int order_item_id
        text product_id FK
        text seller_id FK
        text category
        double item_revenue
        double freight_value
    }
    fct_payments {
        text order_id FK
        int payment_sequential
        text payment_type
        double payment_value
    }
    fct_reviews {
        text review_id
        text order_id FK
        int review_score
    }
    dim_products {
        text product_id PK
        text category
        int weight_g
    }
    dim_sellers {
        text seller_id PK
        text state
        double lat
        double lng
    }
```

Raw-layer relationships (enforced by the raw checks): `orders.customer_id → customers`,
`order_items.order_id → orders`, `order_items.product_id → products`,
`order_items.seller_id → sellers`, `order_payments.order_id → orders`,
`order_reviews.order_id → orders`, `products.product_category_name → category_translation`
(2 categories missing; translated in staging). Full column list:
[data_dictionary.md](data_dictionary.md).

## One implementation, two runtimes

The metric layer (`olist.metrics`), the browser-safe analytics (`olist.analytics.cohorts`,
`rfm`, `sellers`, `delivery`), the experiment simulator and the scenarios use only pandas and
numpy. The pipeline uses them to write `results/`; the dashboard imports the same modules,
from `src/` under `streamlit run` and from copies in the stlite bundle on GitHub Pages
(`scripts/build_site.py: APP_MODULES`). A test fails if the dashboard imports any other
`olist` module or anything heavy (SciPy, scikit-learn, LightGBM, DuckDB). Models that need
SciPy or LightGBM (CLV, the delivery model, the churn model) run in the pipeline, and the
dashboard shows their committed results, labelled "full data".

## Reproducibility

- Inputs: the 9 CSVs are fetched from a pinned commit of Olist's public repository and
  checked against SHA-256 checksums (`scripts/fetch_data.py`).
- Every generated JSON carries `run_metadata` (library versions, seed, a fingerprint of the
  raw files) but no timestamp, so a rebuild from the same code and data on the same machine
  is byte-identical (verified by running the analysis twice); on another CPU, optimiser
  outputs (the CLV fits) can differ in the 6th-7th significant digit. The git commit records
  when and from which code.
- Seeds: `olist.config.RANDOM_SEED`; LightGBM runs single-threaded and deterministic; inputs
  are loaded in a fixed order because resampling and bagging depend on row order.

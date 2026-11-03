# Olist e-commerce analytics

[![CI](https://github.com/srini2k03/olist-ecommerce-analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/srini2k03/olist-ecommerce-analytics/actions/workflows/ci.yml)
[![Build data](https://github.com/srini2k03/olist-ecommerce-analytics/actions/workflows/build-data.yml/badge.svg)](https://github.com/srini2k03/olist-ecommerce-analytics/actions/workflows/build-data.yml)
[![Pages](https://github.com/srini2k03/olist-ecommerce-analytics/actions/workflows/pages.yml/badge.svg)](https://github.com/srini2k03/olist-ecommerce-analytics/actions/workflows/pages.yml)

This project runs SQL, statistics, segmentation and a churn model on the **real** Olist Brazilian e-commerce dataset: 99,441 orders across 9 tables, from Sept 2016 to Oct 2018. It ends with a Streamlit dashboard and a one-page [executive summary](EXECUTIVE_SUMMARY.md).

- **Live dashboard:** https://srini2k03.github.io/olist-ecommerce-analytics/app/ runs Streamlit entirely in your browser via [stlite](https://github.com/whitphx/stlite), with no server. The first visit takes 10 to 30 s while Python starts.
- **Rendered notebooks and summary:** https://srini2k03.github.io/olist-ecommerce-analytics/

## Key findings

Every number below comes from executed code on the real data. The SQL snapshots are in [`results/sql/`](results/sql), and the notebook outputs are in [`notebooks/`](notebooks).

| # | Finding | Evidence |
|---|---|---|
| 1 | **Growth has stalled.** Jan–Aug 2018 GMV was +140% on the same months of 2017, but mean month-on-month growth from Feb to Aug 2018 was **−1.0%**. Monthly GMV has been flat at R$ 1.0–1.16M. | SQL 01, [01_eda](notebooks/01_eda.ipynb) |
| 2 | **Customers almost never return.** Only **3.0%** of 94,707 customers ever place a second order. Month-1 retention averages **0.48%**, and **30%** of second orders are placed the same day as the first (split baskets, not loyalty). | SQL 02, 10, 12 |
| 3 | **Late delivery destroys reviews.** Late orders average **2.27★**, against **4.29★** for on-time orders (−2.02★, 95% bootstrap CI −2.06 to −1.98; Cliff's δ = −0.64, large). With category, state, basket and value controls, the odds ratio for a higher score is **0.068** (95% CI 0.063–0.074). | [02_stats](notebooks/02_stats.ipynb) Part A |
| 4 | **Lateness is regional.** The late rate is **21.4%** in Alagoas against **4.5%** in São Paulo. The single lane **SP → RJ** accounts for **17.6% of all late deliveries**. | SQL 09, 13 |
| 5 | **Value sits in lapsed one-time buyers.** "At-risk big spenders" are **12.4% of customers but 38.6% of revenue** (R$ 515 average, about 298 days since their last order). They also have the lowest mean review of any segment (3.95). | [03_segmentation](notebooks/03_segmentation.ipynb) |
| 6 | **Repeat purchase is barely predictable, and we say so.** With a 1.18% base rate, LightGBM reaches ROC-AUC 0.583 and PR-AUC 0.020 on a later, held-out time window. **Targeting the top 10% of customers ranked by the model captures 19.6% of those who actually bought again within 180 days (1.96x lift), against 15.4% for a recency rule.** Logistic regression is statistically tied. | [04_churn](notebooks/04_churn.ipynb) |
| 7 | **Simulated A/B test, calibrated to real baselines:** free shipping on orders from R$ 150. In the simulation it lifts the multi-item rate (+1.43 pp, p = 0.006, P(B>A) = 99.7%), but the freight subsidy costs **R$ 9.60 per order** against about **R$ 1.44** of extra GMV, because 26% of orders already qualify. Do not ship it as designed. | 02_stats Part B (**simulated**) |

<p align="center">
  <img src="results/figures/delay_vs_review.png" width="48%" alt="Review score by delivery delay">
  <img src="results/figures/cohort_heatmap.png" width="48%" alt="Cohort retention heatmap">
  <img src="results/figures/segments.png" width="48%" alt="Segments: customers vs revenue">
  <img src="results/figures/churn_lift_gain.png" width="48%" alt="Lift and cumulative gain">
</p>

## What is in the repository

```
sql/                     16 documented queries (DuckDB and Postgres, same SQL)
results/sql/             snapshot of every query's output on the real data
results/figures/         README and Pages figures, saved by the notebooks
results/metrics/         JSON metrics written by the notebooks (stats, segments, churn)
notebooks/               01_eda, 02_stats, 03_segmentation, 04_churn (executed, outputs committed)
src/olist/               reusable package
  db.py, schema.py         portable DDL + loaders for DuckDB and Postgres
  sql_runner.py            discover/run/snapshot queries, cross-engine comparison
  warehouse.py             order-level fact table for Python analysis
  stats/                   effect sizes, bootstrap, power, SRM, Bayesian A/B, peeking
  segmentation.py          RFM, k diagnostics, rule-based cluster naming
  model/                   point-in-time features, models, lift/gain, SHAP reasons
  app_data.py              builds the small app artefacts
  synthetic.py             schema-faithful synthetic fixture (tests only)
app/streamlit_app.py     dashboard entry point; app/data/ holds its prebuilt ~4 MB artefacts
scripts/                 fetch_data, build_warehouse, run_sql, build_app_data, smoke_app, build_site
tests/                   pytest: SQL, stats, leakage, models, segmentation, app
.github/workflows/       ci.yml, build-data.yml, pages.yml
```

## Data and licence

- **Dataset:** [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), licensed **CC BY-NC-SA 4.0** (attribution, non-commercial, share-alike). The raw CSVs are **not committed**. `scripts/fetch_data.py` downloads them at build time:
  1. through the Kaggle API, when `KAGGLE_USERNAME`/`KAGGLE_KEY` are set;
  2. otherwise from Olist's own public repository [`olist/work-at-olist-data`](https://github.com/olist/work-at-olist-data), **pinned to commit `d9e4980`**.

  Every file is checked against a SHA-256 checksum. A Kaggle file that differs from the pinned bytes is replaced from the mirror, so every run analyses identical data.
- **Derived artefacts** (aggregated SQL outputs, figures, the per-customer scores in `app/data`) are shared under the same CC BY-NC-SA 4.0 terms.
- **Code:** MIT ([LICENSE](LICENSE)).

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
make setup          # pip install -r requirements-dev.txt -e .
make warehouse      # fetch + verify the 9 CSVs, build data/olist.duckdb
make sql            # run the 16 queries, write results/sql/*.csv
make notebooks      # execute all notebooks in place with nbconvert
make app-data       # rebuild app/data
make test lint      # pytest + ruff
make app            # streamlit run app/streamlit_app.py
```

**Postgres.** Set `OLIST_PG_DSN=postgresql://user:pass@host/db`, run `make postgres` to load the same tables, then run `make sql-compare` to execute every query on both engines and require identical results.

## SQL library (`sql/`)

Each file starts with a header comment giving its **business question**, **method**, **expected output** and **dialect notes**. Together the queries use JOINs, CTEs and window functions: `LAG`, `LEAD`, `NTILE`, `RANK`, `PERCENT_RANK`, `ROW_NUMBER`, running `SUM() OVER`, and `PERCENTILE_CONT ... WITHIN GROUP`.

| # | Query | # | Query |
|---|---|---|---|
| 01 | Monthly GMV, MoM/YoY growth, 3-month average | 09 | State delivery SLA (mean/p90 days, late rate) |
| 02 | Cohort retention by month since first order | 10 | Repeat-purchase rate by cohort quarter |
| 03 | RFM scoring (NTILE) and segment shares | 11 | Basket size vs order value and reviews |
| 04 | Delivery delay buckets vs review score | 12 | Time to second order |
| 05 | Seller performance league | 13 | Late-delivery hot-spot lanes |
| 06 | Category Pareto | 14 | Review sentiment proxies (Portuguese keywords) |
| 07 | Payment mix and instalments | 15 | Customer value deciles |
| 08 | Freight-to-price ratio by category | 16 | Purchase timing (weekday × hour) |

**Portability.** The same text runs on DuckDB and Postgres. The queries avoid engine-specific functions:
- day differences use `CAST(a AS DATE) - CAST(b AS DATE)` rather than DuckDB's `date_diff`;
- months come from `CAST(date_trunc('month', ts) AS DATE)`;
- `EXTRACT` results are cast to `INTEGER`, because Postgres returns `NUMERIC`;
- there is no `ROUND(double, n)`, which Postgres lacks; rounding happens in the runner;
- keyword matching uses `LOWER(...) LIKE` instead of `ILIKE` or regex.

The runner turns Postgres `NUMERIC` into floats and compares every cell with a 1e-6 relative tolerance. CI runs the comparison on the real data.

## Methods in brief

- **Statistics (02).** Mann-Whitney U and Welch's t-test; Cliff's delta and Cohen's d, each with a percentile bootstrap 95% CI; a proportional-odds ordinal regression controlling for category, state, basket and value.
- **Simulated A/B test (02, Part B).** Clearly labelled as simulated. The baseline (9.90% multi-item rate) and traffic (222 orders/day) are real 2018 values. It covers:
  - power and sample size (a 10% MDE would take 135 days, so a 15% MDE over 9 weeks was planned);
  - an SRM chi-square check;
  - a two-proportion z-test with CI;
  - a Beta-Binomial posterior with P(B>A) and expected loss;
  - guardrails resampled from real orders;
  - peeking: 20 looks inflate the false-positive rate from 5% to 23%;
  - a novelty-effect simulation.
- **Segmentation (03).** RFM, then a log transform with frequency capped at 3, standardisation and K-Means. k = 5 was chosen from the elbow, the silhouette and business sense. The silhouette's k = 2 maximum only separates repeat buyers. Clusters are named by explicit rules on their profile, and the result is cross-checked against the SQL NTILE RFM.
- **Churn / repeat purchase (04).**
  - **Split:** time-based, with train cutoffs 2017-06-01 and 2017-09-01, test cutoff 2018-03-01, and a 180-day label window.
  - **Point-in-time features:** deliveries and reviews count only if they had happened by the cutoff.
  - **Leakage:** a dedicated "no leakage" section with a timeline figure, plus a unit test that scrambles post-cutoff data.
  - **Models:** a recency rule and logistic regression as baselines, against LightGBM.
  - **Evaluation:** PR-AUC and ROC-AUC with bootstrap CIs, lift and gain charts, SHAP.

### Being candid about the churn model

Olist's repeat rate is about 3% overall and about 1.2% in a 180-day window, and the data has no browsing or marketing signals. No model here is strong: ROC-AUC is about 0.58–0.59, and the PR-AUC confidence intervals of LightGBM and logistic regression overlap. The model is still useful for ranking **cheap** retention actions, since its top decile holds about 2x the base rate. It should not be used to justify expensive per-customer offers, and an uplift holdout test should come before any incentive spend.

## The dashboard

`app/streamlit_app.py` has the following views:
- KPI cards;
- monthly GMV trend;
- cohort retention heatmap;
- segment explorer;
- delivery-delay impact (by bucket and by state);
- **churn-risk lookup by `customer_unique_id`**, showing P(repeat in 180 days), decile, segment and the top-3 SHAP reasons.

It reads only `app/data/` (about 4 MB of parquet and JSON, built by `scripts/build_app_data.py` and committed by the `build-data` workflow), so it starts quickly and needs no ML libraries at runtime.

<p align="center"><img src="results/figures/app_churn.png" width="80%" alt="Churn-risk lookup in the dashboard"></p>

### Deploy the dashboard

**GitHub Pages (live, the default).** `scripts/build_site.py` copies the app and its prebuilt data into `site/app/` with a [stlite](https://github.com/whitphx/stlite) loader, which runs Streamlit on Pyodide in the browser. Before publishing, the Pages workflow renders it in headless Chromium (`scripts/smoke_browser_app.py`) and fails if it does not load.

**Streamlit Community Cloud (optional).** The repository also works as a normal Streamlit app:

1. `requirements.txt` sits **at the repository root** and contains only the app's runtime dependencies (streamlit, pandas, pyarrow, altair). The full analysis stack is in `requirements-dev.txt`.
2. The entry point is **`app/streamlit_app.py`**.
3. At share.streamlit.io, sign in with GitHub, click **Create app**, pick this repo and branch `main`, and set the main file path to `app/streamlit_app.py`. Under Advanced settings, choose Python 3.12 or later.

## CI/CD

| Workflow | What it proves or does |
|---|---|
| `ci.yml` | It has four jobs. **lint-test:** ruff (lint and format) and pytest on a synthetic fixture, including Streamlit `AppTest` on the committed artefacts. **sql-duckdb-postgres:** fetches the real data, loads DuckDB and a Postgres 16 service container, runs all 16 queries on both, requires identical results, and checks that the committed `results/sql` snapshots match a fresh run. **notebooks:** executes all four notebooks on real data with nbconvert. **streamlit-smoke:** `streamlit run` driven by Playwright Chromium, using only `requirements.txt`. |
| `build-data.yml` | Fetches the real data, rebuilds SQL snapshots, executed notebooks, metrics, figures, app screenshots and `app/data`, and commits any changes. |
| `pages.yml` | Renders the executed notebooks to HTML together with the executive summary and the README figures, and publishes them to `gh-pages` with `peaceiris/actions-gh-pages@v4`. |

## Honesty notes

- All results were produced locally on the real, checksum-verified dataset (Olist commit `d9e4980`) on 2026-11-03. CI recomputes them on every push.
- **The A/B test is simulated.** Only its inputs are real. The notebook labels it as simulated in every heading.
- The review-text "sentiment" uses keyword proxies, not a trained classifier.
- The causal reading of the late-delivery effect rests on observational data plus controls. A promise-date experiment is the proper confirmation.

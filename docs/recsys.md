# Fashion recommender: design note (phase 1)

A production-style, two-stage recommender (candidate retrieval, then ranking) lives
in `src/recsys/` (imported as `recsys`). Phase 1, described here, covers the data
pipeline, the time-based split with leakage checks, the baselines and the metrics.
Everything is tested on synthetic data; **no real-data results exist yet**.

## Why not Olist, and why H&M

Collaborative filtering learns from customers who interact with several items over
time. Olist has almost none of that signal, as the repo's own SQL snapshots show:

| Olist fact | Value | Source |
|---|---|---|
| Customers who ever place a second order | 2,875 of 94,707 (**3.04%**) | `results/sql/10_repeat_purchase_rate.csv` |
| ... within 90 days | 2.08% | same |
| Orders with a single item | 88,438 of 98,199 (**90.1%**) | `results/sql/11_basket_size.csv` |

With ~97% one-time buyers and one item per order, a user-item matrix is almost
empty, so any "recommender" would collapse to category popularity.

The [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)
competition data has what Olist lacks: about 31.8M transactions from about 1.37M
customers over two years (Sep 2018 to Sep 2020), about 105K articles with rich
metadata (product code, type, colour, department, garment group), customer age and
frequent repeat buying. The competition's task (predict the 12 articles each customer
buys in the next 7 days, scored by MAP@12) is a natural offline benchmark.

The competition rules do not allow redistributing the data, so it is never committed:
everything lives under the gitignored `data/hm/`.

## Pipeline

```
python -m recsys fetch      # Kaggle CLI -> data/hm/raw/*.csv + manifest.json, then data/hm/parquet/
python -m recsys subset     # last 16 weeks, 25% of customers -> data/hm/subset/
python -m recsys split      # train / validation / test -> data/hm/split/ + split_report.json
python -m recsys baselines  # fit on train, score validation -> results/recsys/baselines_validation.json
python -m recsys --synthetic  # all of the above on generated data, under data/hm/synthetic/
```

`make recsys-data` runs the first three, `make recsys-baselines` the fourth and
`make recsys-synthetic` the synthetic run. Data paths can be overridden with
`RECSYS_DATA_DIR`, `RECSYS_RAW_DIR` and `RECSYS_RESULTS_DIR` (or `--data-dir`,
`--results-dir`).

**Fetch** (`recsys/data/fetch_hm.py`) downloads only `transactions_train.csv`,
`articles.csv` and `customers.csv` (not the ~30 GB of images) with the Kaggle CLI.
It needs `KAGGLE_USERNAME`/`KAGGLE_KEY` (or `~/.kaggle/kaggle.json`) from an account
that has accepted the competition rules, and fails with a clear message otherwise. A
manifest records each file's size and SHA-256. Conversion streams the 3.5 GB
transactions file in 2M-row chunks and writes parquet with explicit dtypes:
`customer_id` (a 64-character hex string) becomes an int64 (its last 16 hex digits),
which cuts memory several-fold; collisions are checked on the full customer table and
`customer_id_map.parquet` keeps the original ids. `article_id` is int32, `t_dat` a
day-resolution datetime, `price` float32 and `sales_channel_id` int8.

**Subset** (`recsys/data/subset.py`): the last 16 weeks, then a 25% sample of the
customers active in them (both configurable), keeping all their transactions and every
article and customer row those transactions reference. The sample is a seeded hash of
the customer id, so a customer's membership does not depend on the window length. At
the real data's volume this should leave on the order of a million transactions; the
exact counts are logged and written to `data/hm/subset/manifest.json` on the real run.

## Split and leakage

Weeks are counted back from the last transaction date (the competition's hidden test
set is the 7 days after the training data ends):

* **test** = the final week, **validation** = the week before, **train** = everything
  earlier (all three configurable with `--test-weeks`, `--val-weeks`, `--train-weeks`).
* Validation is scored with models fit on train; test with models fit on train +
  validation. Popularity counts, co-occurrence pairs, user histories and the
  popularity percentiles used by the metrics all come from that same history.

Leakage checks (`recsys/data/split.py`, all unit-tested) run on every split and before
every evaluation: no training row on or after the train cutoff; each target split stays
inside its window; any history used to score a split is strictly before that split's
start. `cold_start_report` counts target users and articles never seen in the history;
those users get each baseline's popularity fallback, and later phases will add a proper
cold-start path.

## Baselines

All share `fit(train_df)` and `recommend(user_ids, k) -> {user: [article, ...]}` and
return `k` distinct articles per user, with deterministic tie-breaking:

| Baseline | Idea |
|---|---|
| `popularity_1w` | Most purchased articles in the last training week (filled from all-time popularity). |
| `age_popularity_4w` | Popularity in the user's age bucket over the last 4 weeks; unknown age falls back to global. |
| `item_cooccurrence` | Articles bought by the same customer within 7 days of each other; recommends neighbours of the user's 10 most recent articles (recency-weighted), filled with popularity. |
| `repurchase` | The user's own articles from the last 8 weeks by recency, then frequency, filled with popularity. |

## Metrics

`recsys/evaluate/metrics.py`, evaluated over users with at least one purchase in the
target week, at K = 12, 50 and 100:

* **MAP@K** exactly as in the competition: `AP@K = sum_{i<=K} P(i) rel(i) / min(|relevant|, K)`
  (tested against the reference `apk` implementation, including duplicates, truncation
  and users with more than K purchases).
* **Recall@K**, **NDCG@K** (binary relevance), **HitRate@K**.
* **Catalogue coverage** (share of training articles recommended to anyone),
  **popularity bias** (mean popularity percentile of recommended articles) and the
  **Gini** of recommendation counts across the catalogue.
* **95% bootstrap CIs** over users (1,000 resamples, seeded) for the per-user metrics.

`evaluate(recs, ground_truth, ks)` returns a tidy table (`metric, k, value, ci_low,
ci_high, n_users`).

## What comes next

1. Retrieval: a PyTorch two-tower model (customer and article towers over ids and
   metadata) with FAISS approximate nearest-neighbour search, merged with the baseline
   candidates.
2. Ranking: LightGBM LambdaRank over the candidates with user, item and user-item
   features, all built strictly before each split's start.
3. Cold start for new customers (age/popularity) and new articles (metadata tower).
4. Full evaluation on the real data, including a simulated A/B test that reuses
   `olist.stats.abtest` (power, SRM check, z-test).
5. Serving: FastAPI + Docker, a Streamlit demo separate from the Olist dashboard (which
   must stay free of ML dependencies), and CI.

"""Build the small artefacts the Streamlit app reads (``app/data``).

The app never touches the raw CSVs or the warehouse: it reads these files, so
it starts fast on Streamlit Community Cloud. ``scripts/build_app_data.py`` runs
this module, and the ``build-data`` workflow commits the result.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from olist.config import APP_DATA_DIR, RANDOM_SEED, WAREHOUSE_PATH
from olist.model.evaluate import ranking_metrics, reason_codes, shap_matrix
from olist.model.features import build_snapshot, build_training_set
from olist.model.train import make_lgbm, to_lgbm_frame
from olist.segmentation import SEGMENT_ACTIONS, fit_kmeans, name_segments, rfm_table, transform
from olist.warehouse import connect, order_facts, run_sql_file

HORIZON = 180
EVAL_TRAIN_CUTOFFS = ["2017-06-01", "2017-09-01"]
EVAL_TEST_CUTOFF = "2018-03-01"
FINAL_TRAIN_CUTOFFS = ["2017-06-01", "2017-09-01", "2017-12-01", "2018-03-01"]
SCORING_DATE = "2018-09-01"


def _write(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path, index=False, compression="zstd")


def build(out_dir: Path = APP_DATA_DIR, warehouse: Path = WAREHOUSE_PATH) -> dict[str, int]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    con = connect(warehouse)
    orders = order_facts(con)

    # --- SQL-backed views ----------------------------------------------------
    monthly = run_sql_file(con, "01_monthly_revenue_mom")
    cohort = run_sql_file(con, "02_cohort_retention")
    delay = run_sql_file(con, "04_delivery_delay_vs_review")
    sla = run_sql_file(con, "09_state_delivery_sla")
    repeat = run_sql_file(con, "10_repeat_purchase_rate")
    for df in (monthly, cohort):
        for c in df.columns:
            if "month" in c and c != "months_since_first":
                df[c] = pd.to_datetime(df[c]).dt.date.astype(str)
    _write(monthly, out_dir / "monthly.parquet")
    _write(cohort, out_dir / "cohort.parquet")
    _write(delay, out_dir / "delivery_delay.parquet")
    _write(sla, out_dir / "state_sla.parquet")

    # --- KPIs ------------------------------------------------------------------
    window = orders[(orders.purchased_at >= "2017-01-01") & (orders.purchased_at < "2018-09-01")]
    delivered = window[window.delay_days.notna()]
    overall = repeat[repeat.cohort_quarter == "All"].iloc[0]
    kpis = {
        "period": "2017-01 to 2018-08",
        "gmv": float(window.order_value.sum()),
        "orders": int(len(window)),
        "customers": int(window.customer_unique_id.nunique()),
        "aov": float(window.order_value.mean()),
        "avg_review": float(window.review_score.mean()),
        "late_rate": float(delivered.is_late.mean()),
        "repeat_rate": float(overall.repeat_rate),
        "gmv_last_month": float(monthly.gmv.iloc[-1]),
        "mom_last_month": float(monthly.mom_growth.iloc[-1]),
    }

    # --- Segments (same procedure as notebook 03) ------------------------------
    rfm = rfm_table(orders, SCORING_DATE)
    km = fit_kmeans(transform(rfm), 5)
    rfm["cluster"] = km.labels_
    profile = rfm.groupby("cluster").agg(
        customers=("customer_unique_id", "size"),
        recency_days=("recency_days", "mean"),
        frequency=("frequency", "mean"),
        monetary=("monetary", "mean"),
        avg_review=("avg_review", "mean"),
        revenue=("monetary", "sum"),
    )
    names = name_segments(profile)
    profile["segment"] = profile.index.map(names)
    profile["customer_share"] = profile.customers / profile.customers.sum()
    profile["revenue_share"] = profile.revenue / profile.revenue.sum()
    profile["action"] = profile.segment.map(SEGMENT_ACTIONS)
    rfm["segment"] = rfm.cluster.map(names)
    _write(
        profile.sort_values("revenue_share", ascending=False).reset_index(drop=True),
        out_dir / "segments.parquet",
    )
    sample = rfm.sample(4000, random_state=RANDOM_SEED)[
        ["recency_days", "frequency", "monetary", "segment"]
    ]
    _write(sample, out_dir / "segment_sample.parquet")

    # --- Churn model: evaluation (same split as notebook 04) -------------------
    train = build_training_set(orders, EVAL_TRAIN_CUTOFFS, HORIZON)
    test = build_snapshot(orders, EVAL_TEST_CUTOFF, HORIZON)
    X_tr, cats = to_lgbm_frame(train)
    X_te, _ = to_lgbm_frame(test, cats)
    eval_model = make_lgbm().fit(X_tr, train.label)
    test_metrics = ranking_metrics(test.label, eval_model.predict_proba(X_te)[:, 1])

    # --- Churn model: final fit on every fully-labelled snapshot, score today --
    final = build_training_set(orders, FINAL_TRAIN_CUTOFFS, HORIZON)
    X_f, cats_f = to_lgbm_frame(final)
    model = make_lgbm().fit(X_f, final.label)
    now = build_snapshot(orders, SCORING_DATE, HORIZON)
    X_now, _ = to_lgbm_frame(now, cats_f)
    p = model.predict_proba(X_now)[:, 1]
    sv = shap_matrix(model, X_now)
    scores = pd.DataFrame(
        {
            "customer_unique_id": now.customer_unique_id,
            "p_repeat_180d": p.astype("float32"),
            "churn_risk": (1 - p).astype("float32"),
            "repeat_decile": (10 - pd.qcut(p, 10, labels=False, duplicates="drop")).astype("int8"),
            "customer_state": now.customer_state,
            "orders": now.n_orders.astype("int16"),
            "total_spend": now.total_spend.round(2).astype("float32"),
            "recency_days": now.recency_days.astype("int16"),
            "last_category": now.last_category,
            "reasons": reason_codes(sv, X_now, top_n=3),
        }
    )
    scores = scores.merge(rfm[["customer_unique_id", "segment"]], on="customer_unique_id")
    _write(scores, out_dir / "churn_scores.parquet")
    importance = pd.Series(np.abs(sv).mean(axis=0), index=X_now.columns).sort_values()
    model_card = {
        "horizon_days": HORIZON,
        "scoring_date": SCORING_DATE,
        "final_train_cutoffs": FINAL_TRAIN_CUTOFFS,
        "final_train_rows": int(len(final)),
        "final_train_positives": int(final.label.sum()),
        "eval_train_cutoffs": EVAL_TRAIN_CUTOFFS,
        "eval_test_cutoff": EVAL_TEST_CUTOFF,
        "test_metrics": test_metrics,
        "test_rows": int(len(test)),
        "test_positives": int(test.label.sum()),
        "mean_abs_shap": importance[::-1].to_dict(),
    }
    (out_dir / "kpis.json").write_text(json.dumps(kpis, indent=2))
    (out_dir / "model_card.json").write_text(json.dumps(model_card, indent=2))
    con.close()
    return {p.name: p.stat().st_size for p in sorted(out_dir.iterdir())}

"""Build the compact bundle the dashboard reads (``app/data``).

The dashboard never touches the raw CSVs or the warehouse. It reads:

- ``orders.parquet``: every order (one row each) with the fct_orders columns the metric layer
  needs, ids replaced by integer codes and low-cardinality text stored as categories;
- ``items.parquet``: one row per item (order row index, seller code, category, price,
  freight), so category and seller filters attribute revenue exactly;
- ``sellers.parquet``: seller code -> seller_id, state, city;
- small JSON summaries: the metric spec, the data-quality report and the analytics results;
- the K-Means segments and churn scores of notebooks 03/04 (full data, not filterable).

``scripts/build_app_data.py`` runs this module; the ``build-data`` workflow commits the result.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from olist import metrics
from olist.config import (
    ANALYTICS_DIR,
    APP_DATA_DIR,
    DQ_DIR,
    METRICS_SPEC,
    RANDOM_SEED,
    WAREHOUSE_PATH,
)
from olist.model.evaluate import ranking_metrics, reason_codes, shap_matrix
from olist.model.features import build_snapshot, build_training_set
from olist.model.train import make_lgbm, to_lgbm_frame
from olist.segmentation import SEGMENT_ACTIONS, fit_kmeans, name_segments, rfm_table, transform
from olist.warehouse import connect, order_facts

HORIZON = 180
EVAL_TRAIN_CUTOFFS = ["2017-06-01", "2017-09-01"]
EVAL_TEST_CUTOFF = "2018-03-01"
FINAL_TRAIN_CUTOFFS = ["2017-06-01", "2017-09-01", "2017-12-01", "2018-03-01"]
SCORING_DATE = "2018-09-01"


def _write(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path, index=False, compression="zstd")


ORDER_COLUMNS = {
    # column: dtype in the bundle
    "is_valid": "int8",
    "is_cancelled": "int8",
    "is_delivered": "int8",
    "is_late": "float32",
    "purchase_date": "datetime64[ms]",
    "purchase_month": "datetime64[ms]",
    "cohort_month": "datetime64[ms]",
    "customer_order_seq": "float32",
    "customer_state": "category",
    "seller_state": "category",
    "category": "category",
    "order_status": "category",
    "main_payment_type": "category",
    "n_items": "float32",
    "installments": "float32",
    "item_revenue": "float64",
    "freight_value": "float64",
    "payment_value": "float64",
    "delivery_days": "float32",
    "estimated_days": "float32",
    "delay_days": "float32",
    "review_score": "float32",
    "distance_km": "float32",
}
OBSOLETE = (
    "monthly.parquet",
    "cohort.parquet",
    "delivery_delay.parquet",
    "state_sla.parquet",
    "kpis.json",
)
ANALYTICS_FILES = (
    "findings.json",
    "clv.json",
    "delivery_model.json",
    "seller_ranking.json",
    "ab_test.json",
    "metrics_snapshot.json",
)


def build_bundle(out_dir: Path, con) -> dict[str, int]:
    """orders.parquet, items.parquet and sellers.parquet from the marts."""
    fo = con.execute("SELECT * FROM fct_orders ORDER BY purchased_at, order_id").df()
    it = con.execute(
        "SELECT order_id, order_item_id, seller_id, category, item_revenue, freight_value "
        "FROM fct_order_items ORDER BY order_id, order_item_id"
    ).df()
    sellers = con.execute("SELECT seller_id, state, city FROM dim_sellers ORDER BY seller_id").df()
    seller_code = pd.Series(np.arange(len(sellers), dtype="int16"), index=sellers.seller_id)
    customer_code, _ = pd.factorize(fo["customer_unique_id"])

    orders = pd.DataFrame({c: fo[c] for c in ORDER_COLUMNS})
    for c, dtype in ORDER_COLUMNS.items():
        if dtype.startswith("datetime"):
            orders[c] = pd.to_datetime(orders[c]).astype(dtype)
        elif dtype == "category":
            orders[c] = orders[c].astype("string").astype("category")
        else:
            orders[c] = pd.to_numeric(orders[c], errors="coerce").astype(dtype)
    orders.insert(0, "customer_unique_id", customer_code.astype("int32"))
    # Whole kilometres are plenty for distance bands and compress far better.
    orders["distance_km"] = orders["distance_km"].round(0).astype("float32")
    orders["main_seller_id"] = (
        fo["main_seller_id"].map(seller_code).fillna(-1).astype("int16").to_numpy()
    )
    row = pd.Series(np.arange(len(fo), dtype="int32"), index=fo["order_id"])
    items = pd.DataFrame(
        {
            "order_idx": it["order_id"].map(row).astype("int32"),
            "seller_id": it["seller_id"].map(seller_code).astype("int16"),
            "category": it["category"].astype("string").astype("category"),
            "item_revenue": it["item_revenue"].astype("float64"),
            "freight_value": it["freight_value"].astype("float64"),
        }
    ).sort_values(["order_idx", "seller_id"], kind="stable")
    sellers.insert(0, "code", seller_code.to_numpy())
    _write(orders, out_dir / "orders.parquet")
    _write(items, out_dir / "items.parquet")
    _write(sellers, out_dir / "sellers.parquet")
    return {"orders": len(orders), "items": len(items), "sellers": len(sellers)}


def copy_summaries(out_dir: Path, analytics_dir: Path = ANALYTICS_DIR, dq_dir: Path = DQ_DIR):
    """Metric spec (as JSON), data-quality report and the analytics results the app shows."""
    (out_dir / "metrics.json").write_text(json.dumps(metrics.load_spec(METRICS_SPEC), indent=1))
    dq = json.loads((Path(dq_dir) / "report.json").read_text())
    (out_dir / "data_quality.json").write_text(json.dumps(dq, indent=1))
    for name in ANALYTICS_FILES:
        src = Path(analytics_dir) / name
        if src.exists():
            shutil.copy2(src, out_dir / name)
    for extra in ("clv_by_segment.csv", "retention_curve.csv"):
        src = Path(analytics_dir) / extra
        if src.exists():
            _write(pd.read_csv(src), out_dir / extra.replace(".csv", ".parquet"))


def build(
    out_dir: Path = APP_DATA_DIR, warehouse: Path = WAREHOUSE_PATH, with_models: bool = True
) -> dict[str, int]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in OBSOLETE:
        (out_dir / name).unlink(missing_ok=True)
    con = connect(warehouse)
    build_bundle(out_dir, con)
    copy_summaries(out_dir)
    if not with_models:
        con.close()
        return {p.name: p.stat().st_size for p in sorted(out_dir.iterdir())}
    orders = order_facts(con)

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
            "repeat_decile": (10 - pd.qcut(p, 10, labels=False, duplicates="drop")).astype("int8"),
            "customer_state": now.customer_state.astype("category"),
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
    (out_dir / "model_card.json").write_text(json.dumps(model_card, indent=2))
    con.close()
    return {p.name: p.stat().st_size for p in sorted(out_dir.iterdir())}

"""Delivery-time and late-delivery prediction at purchase time, validated on later months.

Question: when an order is placed, can we predict how long it will take and whether it will
miss the promised date better than (a) Olist's own promise and (b) the historical average
of its seller-state -> customer-state lane?

Leakage rules: every feature is known when the order is placed. Seller and lane history
only use orders *delivered before this order's purchase time* (merge_asof on delivery
time), and the demand feature counts orders placed in the previous 7 days.

Split (by purchase date): train 2017, validation Jan-Mar 2018 (early stopping), test
Apr-Jul 2018. August 2018 is excluded because late August orders may not have been
delivered before the extract ended (right-censoring would bias the test towards fast
deliveries); the censoring share is reported.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from olist.config import RANDOM_SEED

SPLITS = {
    "train": ("2017-01-01", "2018-01-01"),
    "valid": ("2018-01-01", "2018-04-01"),
    "test": ("2018-04-01", "2018-08-01"),
}
CATEGORICAL = ["customer_state", "seller_state", "category", "lane"]
NUMERIC = [
    "estimated_days",
    "distance_km",
    "same_state",
    "n_items",
    "n_sellers",
    "item_revenue",
    "freight_value",
    "freight_ratio",
    "weight_g",
    "volume_cm3",
    "purchase_month",
    "purchase_dow",
    "purchase_hour",
    "orders_prev_7d",
    "seller_past_orders",
    "seller_past_avg_days",
    "seller_past_late_rate",
    "lane_past_orders",
    "lane_past_avg_days",
    "lane_past_late_rate",
]
LGBM_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": RANDOM_SEED,
    "n_jobs": 1,
    "deterministic": True,
    "force_col_wise": True,
    "verbose": -1,
}


def _history(events: pd.DataFrame, queries: pd.DataFrame, key: str, prefix: str) -> pd.DataFrame:
    """Point-in-time aggregates of past deliveries per ``key``: for each query (key,
    purchased_at), the count, mean delivery days and late rate of the key's orders delivered
    strictly before purchased_at."""
    ev = events.sort_values("delivered_at")[[key, "delivered_at", "delivery_days", "is_late"]]
    ev = ev.assign(
        _n=ev.groupby(key).cumcount() + 1,
        _days=ev.groupby(key)["delivery_days"].cumsum(),
        _late=ev.groupby(key)["is_late"].cumsum(),
    )
    q = queries[["order_id", key, "purchased_at"]].sort_values("purchased_at")
    q = q.assign(_t=q["purchased_at"] - pd.Timedelta(microseconds=1))
    m = pd.merge_asof(
        q,
        ev.rename(columns={"delivered_at": "_t"}).drop(columns=["delivery_days", "is_late"]),
        on="_t",
        by=key,
        direction="backward",
    )
    n = m["_n"].fillna(0)
    return pd.DataFrame(
        {
            "order_id": m["order_id"],
            f"{prefix}_past_orders": n,
            f"{prefix}_past_avg_days": (m["_days"] / n).where(n > 0),
            f"{prefix}_past_late_rate": (m["_late"] / n).where(n > 0),
        }
    )


def build_features(fct_orders: pd.DataFrame) -> pd.DataFrame:
    """Feature table for every valid order (targets are NaN where not delivered)."""
    o = fct_orders[fct_orders["is_valid"] == 1].sort_values("order_id").reset_index(drop=True)
    o["purchased_at"] = pd.to_datetime(o["purchased_at"])
    o["delivered_at"] = pd.to_datetime(o["delivered_at"])
    for c in ("delivery_days", "is_late", "estimated_days"):
        o[c] = pd.to_numeric(o[c], errors="coerce").astype(float)
    o["lane"] = o["seller_state"].fillna("?") + ">" + o["customer_state"].fillna("?")
    o["freight_ratio"] = o["freight_value"] / o["item_revenue"].replace(0, np.nan)
    o["weight_g"] = o["main_product_weight_g"]
    o["volume_cm3"] = o["main_product_volume_cm3"]
    o["purchase_month"] = o["purchased_at"].dt.month
    o["purchase_dow"] = o["purchased_at"].dt.dayofweek
    o["purchase_hour"] = o["purchased_at"].dt.hour
    allo = fct_orders[["purchased_at"]].assign(
        purchased_at=lambda x: pd.to_datetime(x.purchased_at)
    )
    times = np.sort(allo["purchased_at"].to_numpy())
    t = o["purchased_at"].to_numpy()
    o["orders_prev_7d"] = np.searchsorted(times, t, side="left") - np.searchsorted(
        times, t - np.timedelta64(7, "D"), side="left"
    )
    events = o[o["is_delivered"] == 1].assign(seller=o["main_seller_id"])
    o["seller"] = o["main_seller_id"]
    for key, prefix in (("seller", "seller"), ("lane", "lane")):
        h = _history(events, o, key, prefix)
        o = o.merge(h, on="order_id", how="left")
    for c in CATEGORICAL:
        o[c] = o[c].astype("category")
    return o


def split(features: pd.DataFrame) -> dict[str, pd.DataFrame]:
    d = features[features["is_delivered"] == 1]
    out = {}
    for name, (lo, hi) in SPLITS.items():
        m = (d["purchased_at"] >= lo) & (d["purchased_at"] < hi)
        out[name] = d[m]
    return out


def censoring_report(features: pd.DataFrame) -> dict:
    """Share of valid orders still undelivered, by purchase month, for the last months."""
    f = features.assign(month=features["purchased_at"].dt.to_period("M").astype(str))
    tail = f[f["purchased_at"] >= "2018-05-01"]
    rate = 1 - tail.groupby("month")["is_delivered"].mean()
    return {k: round(float(v), 4) for k, v in rate.items()}


def _bootstrap(metric, y, p, n_boot: int = 300, seed: int = RANDOM_SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        try:
            vals.append(metric(y[i], p[i]))
        except ValueError:
            continue
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def _regression_metrics(y, pred) -> dict:
    from sklearn.metrics import mean_absolute_error

    y, pred = np.asarray(y, float), np.asarray(pred, float)
    mae = float(mean_absolute_error(y, pred))
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    return {
        "mae": mae,
        "mae_ci": _bootstrap(lambda a, b: float(np.mean(np.abs(a - b))), y, pred),
        "rmse": rmse,
        "bias": float(np.mean(pred - y)),
        "within_3_days": float(np.mean(np.abs(y - pred) <= 3)),
    }


def _classification_metrics(y, score) -> dict:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    y, score = np.asarray(y, int), np.asarray(score, float)
    k = max(1, int(round(0.1 * len(y))))
    top = np.argsort(-score, kind="stable")[:k]  # ties (lane rates) break by row order
    return {
        "pr_auc": float(average_precision_score(y, score)),
        "pr_auc_ci": _bootstrap(average_precision_score, y, score),
        "roc_auc": float(roc_auc_score(y, score)),
        "brier": float(brier_score_loss(y, np.clip(score, 0, 1))),
        "precision_top10pct": float(y[top].mean()),
        "recall_top10pct": float(y[top].sum() / max(y.sum(), 1)),
        "base_rate": float(y.mean()),
    }


SEASONAL = ["purchase_month", "orders_prev_7d"]
CANDIDATES = {
    # name: (regression objective, drop these features)
    "l2_all_features": ("regression", []),
    "l1_all_features": ("regression_l1", []),
    "l1_no_seasonal_level": ("regression_l1", SEASONAL),
}
CLF_CANDIDATES = {"all_features": [], "no_seasonal_level": SEASONAL}


def _fit(kind: str, objective: str, X_tr, y_tr, X_va=None, y_va=None, n_estimators=None):
    import lightgbm as lgb

    params = dict(LGBM_PARAMS)
    if n_estimators is not None:
        params["n_estimators"] = n_estimators
    model = (lgb.LGBMRegressor if kind == "reg" else lgb.LGBMClassifier)(
        objective=objective, **params
    )
    if X_va is not None:
        model.fit(
            X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(100, verbose=False)]
        )
    else:
        model.fit(X_tr, y_tr)
    return model


def _lane_baselines(history: pd.DataFrame, target: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    lane_days = history.groupby("lane", observed=True)["delivery_days"].mean().rename(index=str)
    lane_late = history.groupby("lane", observed=True)["is_late"].mean().rename(index=str)
    lane = target["lane"].astype(str)
    days = pd.to_numeric(lane.map(lane_days), errors="coerce")
    late = pd.to_numeric(lane.map(lane_late), errors="coerce")
    return (
        days.fillna(float(history["delivery_days"].mean())).to_numpy(float),
        late.fillna(float(history["is_late"].mean())).to_numpy(float),
    )


def evaluate(fct_orders: pd.DataFrame) -> dict:
    """Select on validation, refit on train+validation, evaluate once on test."""
    from sklearn.metrics import average_precision_score

    feats = build_features(fct_orders)
    parts = split(feats)
    tr, va, te = parts["train"], parts["valid"], parts["test"]
    trva = pd.concat([tr, va])
    cols = CATEGORICAL + NUMERIC

    # --- model selection on the validation window (test untouched) -----------------
    selection = {"regression": {}, "classification": {}}
    fitted: dict[str, object] = {}
    for name, (objective, drop) in CANDIDATES.items():
        use = [c for c in cols if c not in drop]
        m = _fit("reg", objective, tr[use], tr["delivery_days"], va[use], va["delivery_days"])
        fitted[f"reg:{name}"] = (m, use)
        mae = float(np.mean(np.abs(m.predict(va[use]) - va["delivery_days"])))
        selection["regression"][name] = {"valid_mae": mae, "best_iteration": m.best_iteration_}
    for name, drop in CLF_CANDIDATES.items():
        use = [c for c in cols if c not in drop]
        y_tr, y_va = tr["is_late"].astype(int), va["is_late"].astype(int)
        m = _fit("clf", "binary", tr[use], y_tr, va[use], y_va)
        fitted[f"clf:{name}"] = (m, use)
        ap = float(average_precision_score(y_va, m.predict_proba(va[use])[:, 1]))
        selection["classification"][name] = {
            "valid_pr_auc": ap,
            "best_iteration": m.best_iteration_,
        }
    lane_va_days, lane_va_late = _lane_baselines(tr, va)
    selection["regression"]["lane_history_mean (baseline)"] = {
        "valid_mae": float(np.mean(np.abs(lane_va_days - va["delivery_days"])))
    }
    selection["classification"]["lane_history_rate (baseline)"] = {
        "valid_pr_auc": float(average_precision_score(va["is_late"].astype(int), lane_va_late))
    }
    best_reg = min(CANDIDATES, key=lambda k: selection["regression"][k]["valid_mae"])
    best_clf = max(CLF_CANDIDATES, key=lambda k: selection["classification"][k]["valid_pr_auc"])

    # --- refit on train + validation with the selected settings; evaluate on test --
    obj, drop = CANDIDATES[best_reg]
    use_r = [c for c in cols if c not in drop]
    n_r = max(int(selection["regression"][best_reg]["best_iteration"] or 100), 50)
    reg = _fit("reg", obj, trva[use_r], trva["delivery_days"], n_estimators=n_r)
    use_c = [c for c in cols if c not in CLF_CANDIDATES[best_clf]]
    n_c = max(int(selection["classification"][best_clf]["best_iteration"] or 100), 50)
    clf = _fit("clf", "binary", trva[use_c], trva["is_late"].astype(int), n_estimators=n_c)
    pred_days = reg.predict(te[use_r])
    p_late = clf.predict_proba(te[use_c])[:, 1]
    te_lane_days, te_lane_late = _lane_baselines(trva, te)

    y_days, y_late = te["delivery_days"].to_numpy(float), te["is_late"].to_numpy(int)
    # Robustness: the same selected settings and the lane baseline fitted on 2017 only.
    m_r, u_r = fitted[f"reg:{best_reg}"]
    m_c, u_c = fitted[f"clf:{best_clf}"]
    tr_lane_days, tr_lane_late = _lane_baselines(tr, te)
    robustness = {
        "description": "models (early-stopped on validation) and lane baseline fitted on the "
        "2017 training window only, scored on the same test window",
        "regression_mae": {
            "lane_history_mean": float(np.mean(np.abs(tr_lane_days - y_days))),
            "lightgbm": float(np.mean(np.abs(m_r.predict(te[u_r]) - y_days))),
        },
        "classification_pr_auc": {
            "lane_history_rate": float(average_precision_score(y_late, tr_lane_late)),
            "lightgbm": float(average_precision_score(y_late, m_c.predict_proba(te[u_c])[:, 1])),
        },
    }
    imp = pd.Series(clf.booster_.feature_importance("gain"), index=use_c)
    imp_reg = pd.Series(reg.booster_.feature_importance("gain"), index=use_r)
    late_from_reg = (pred_days > te["estimated_days"].to_numpy(float)).astype(int)
    return {
        "splits": {
            k: {"from": a, "to_exclusive": b, "orders": len(parts[k])}
            for k, (a, b) in SPLITS.items()
        },
        "protocol": "candidates compared on the validation window only; the selected setting "
        "and the lane baseline are refitted on train+validation; the test window is scored once",
        "censoring_undelivered_share": censoring_report(feats),
        "features": {"categorical": CATEGORICAL, "numeric": NUMERIC},
        "params": {k: v for k, v in LGBM_PARAMS.items() if k != "verbose"},
        "selection": selection,
        "selected": {
            "regression": best_reg,
            "classification": best_clf,
            "n_estimators": {"regressor": n_r, "classifier": n_c},
        },
        "regression_delivery_days": {
            "olist_promise": _regression_metrics(y_days, te["estimated_days"]),
            "lane_history_mean": _regression_metrics(y_days, te_lane_days),
            "lightgbm": _regression_metrics(y_days, pred_days),
        },
        "classification_late": {
            "lane_history_rate": _classification_metrics(y_late, te_lane_late),
            "lightgbm": _classification_metrics(y_late, p_late),
            "lightgbm_regressor_vs_promise": {
                "precision": float(y_late[late_from_reg == 1].mean())
                if late_from_reg.sum()
                else math.nan,
                "recall": float(late_from_reg[y_late == 1].mean()),
                "flagged_share": float(late_from_reg.mean()),
            },
        },
        "olist_promise_late_rate_test": float(y_late.mean()),
        "robustness_train_only": robustness,
        "importance_gain_classifier": (imp / imp.sum())
        .sort_values(ascending=False)
        .round(4)
        .head(12)
        .to_dict(),
        "importance_gain_regressor": (imp_reg / imp_reg.sum())
        .sort_values(ascending=False)
        .round(4)
        .head(12)
        .to_dict(),
    }

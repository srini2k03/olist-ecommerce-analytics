"""Ranking metrics, lift/gain tables and SHAP reason codes."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from olist.model.features import FEATURE_LABELS


def top_fraction_capture(y: np.ndarray, score: np.ndarray, frac: float = 0.1) -> float:
    """Share of all positives found in the top ``frac`` of customers by score."""
    y = np.asarray(y)
    order = np.argsort(-np.asarray(score), kind="stable")
    k = int(np.ceil(frac * len(y)))
    return float(y[order[:k]].sum() / max(y.sum(), 1))


def ranking_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    y = np.asarray(y)
    if y.size == 0:
        raise ValueError("empty input")
    base = float(y.mean())
    cap10 = top_fraction_capture(y, score, 0.1)
    return {
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
        "base_rate": base,
        "capture_top10": cap10,
        "lift_top10": cap10 / 0.1,
    }


def gains_table(y: np.ndarray, score: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Decile table: response rate, lift and cumulative gain (decile 1 = highest score)."""
    y = np.asarray(y)
    order = np.argsort(-np.asarray(score), kind="stable")
    ys = y[order]
    bins = np.array_split(np.arange(len(ys)), n_bins)
    rows, cum = [], 0
    for i, idx in enumerate(bins, start=1):
        pos = int(ys[idx].sum())
        cum += pos
        rows.append(
            {
                "decile": i,
                "customers": len(idx),
                "positives": pos,
                "response_rate": pos / len(idx),
                "lift": (pos / len(idx)) / y.mean(),
                "cumulative_population": (idx[-1] + 1) / len(ys),
                "cumulative_gain": cum / y.sum(),
            }
        )
    return pd.DataFrame(rows)


def shap_matrix(model, X: pd.DataFrame) -> np.ndarray:
    """SHAP values (log-odds contributions to repeat purchase) for a LightGBM model."""
    import shap

    explainer = shap.TreeExplainer(model)
    vals = explainer.shap_values(X)
    if isinstance(vals, list):  # older shap: one array per class
        vals = vals[1]
    vals = np.asarray(vals)
    if vals.ndim == 3:
        vals = vals[:, :, 1]
    return vals


def _fmt(v) -> str:
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return "unknown"
        return f"{v:,.0f}" if abs(v) >= 10 else f"{v:.2f}".rstrip("0").rstrip(".")
    return str(v)


def reason_codes(shap_vals: np.ndarray, X: pd.DataFrame, top_n: int = 3) -> list[str]:
    """Human-readable top-n reasons per row, ordered by |SHAP|.

    Negative SHAP pushes towards *not* returning, i.e. raises churn risk.
    """
    cols = list(X.columns)
    out = []
    idx = np.argsort(-np.abs(shap_vals), axis=1)[:, :top_n]
    values = X.to_numpy(object)
    for r in range(len(X)):
        parts = []
        for j in idx[r]:
            direction = "raises churn risk" if shap_vals[r, j] < 0 else "lowers churn risk"
            label = FEATURE_LABELS.get(cols[j], cols[j])
            parts.append(f"{label} = {_fmt(values[r, j])} ({direction})")
        out.append("; ".join(parts))
    return out

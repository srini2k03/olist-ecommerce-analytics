"""RFM features and K-Means customer segmentation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from olist.config import RANDOM_SEED

RFM_COLUMNS = ["recency_days", "frequency", "monetary"]


def rfm_table(orders: pd.DataFrame, as_of: str | pd.Timestamp) -> pd.DataFrame:
    """Recency (days), frequency (orders) and monetary (R$) per customer before ``as_of``."""
    as_of = pd.Timestamp(as_of)
    o = orders[orders["purchased_at"] < as_of]
    g = o.groupby("customer_unique_id")
    rfm = pd.DataFrame(
        {
            "recency_days": (as_of - g["purchased_at"].max()).dt.days,
            "frequency": g["order_id"].nunique(),
            "monetary": g["order_value"].sum(),
            "avg_review": g["review_score"].mean(),
            "customer_state": g["customer_state"].last(),
        }
    )
    return rfm.reset_index()


def transform(rfm: pd.DataFrame) -> np.ndarray:
    """Log-transform the skewed columns, then standardise.

    Frequency is capped at 3 before the log: beyond three orders there are too
    few customers to deserve their own region of feature space.
    """
    X = np.column_stack(
        [
            rfm["recency_days"].to_numpy(float),
            np.log1p(rfm["frequency"].clip(upper=3).to_numpy(float)),
            np.log1p(rfm["monetary"].to_numpy(float)),
        ]
    )
    return StandardScaler().fit_transform(X)


def k_diagnostics(
    X: np.ndarray, ks: range = range(2, 10), sample: int = 20_000, seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """Inertia (elbow) on all rows and silhouette on a random sample, per k."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(sample, len(X)), replace=False)
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        rows.append(
            {
                "k": k,
                "inertia": km.inertia_,
                "silhouette": silhouette_score(X[idx], km.labels_[idx]),
                "smallest_cluster_share": np.bincount(km.labels_).min() / len(X),
            }
        )
    return pd.DataFrame(rows)


def fit_kmeans(X: np.ndarray, k: int, seed: int = RANDOM_SEED) -> KMeans:
    return KMeans(n_clusters=k, n_init=20, random_state=seed).fit(X)


SEGMENT_ACTIONS = {
    "Champions": (
        "VIP treatment: early access, loyalty points, referral asks. Protect, don't discount."
    ),
    "At-risk big spenders": "Personal win-back offer on the category they bought, before day 365.",
    "Recent mid-value buyers": (
        "Second-order nudge within 30-90 days: complementary-category bundle."
    ),
    "Recent low-value buyers": "Cheap reactivation: free-shipping-threshold reminder, cross-sell.",
    "One-and-done (lapsed, low value)": (
        "Low-cost channels only (email/newsletter); do not pay to reacquire."
    ),
}


def name_segments(profile: pd.DataFrame) -> dict[int, str]:
    """Map cluster ids to business names from their centroid profile.

    ``profile`` has one row per cluster with mean ``recency_days``, ``frequency``
    and ``monetary``. The rules are relative (medians across clusters), so they
    survive re-fits; any cluster not matched by a specific rule falls back to a
    descriptive name.
    """
    names: dict[int, str] = {}
    rec_med = profile["recency_days"].median()
    mon_med = profile["monetary"].median()
    champions = profile["frequency"].idxmax()
    if profile.loc[champions, "frequency"] >= 1.5:
        names[champions] = "Champions"
    for cid, row in profile.iterrows():
        if cid in names:
            continue
        recent = row["recency_days"] <= rec_med
        big = row["monetary"] >= mon_med
        if recent and big:
            names[cid] = "Recent mid-value buyers"
        elif recent:
            names[cid] = "Recent low-value buyers"
        elif big:
            names[cid] = "At-risk big spenders"
        else:
            names[cid] = "One-and-done (lapsed, low value)"
    # Disambiguate duplicates deterministically (e.g. two "recent" clusters).
    seen: dict[str, int] = {}
    for cid in sorted(names, key=lambda c: -profile.loc[c, "monetary"]):
        n = names[cid]
        seen[n] = seen.get(n, 0) + 1
        if seen[n] > 1:
            names[cid] = f"{n} ({seen[n]})"
    return names

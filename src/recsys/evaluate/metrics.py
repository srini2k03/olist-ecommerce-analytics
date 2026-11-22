"""Offline top-K metrics for recommendations.

Conventions (shared by every metric):

* ``recs`` is a ranked list per user; only its first ``k`` entries count, and a
  repeated article inside the top ``k`` is a miss (it uses a slot but cannot be
  relevant twice). This matches the reference ``apk`` used by the H&M
  competition.
* ``relevant`` is the set of distinct articles the user bought in the target
  window. Users with no relevant articles are not evaluated (``evaluate`` only
  iterates over ground-truth users); a ground-truth user with no
  recommendations scores 0.

MAP@K follows the competition definition:
``AP@K = sum_{i<=K} P(i) * rel(i) / min(|relevant|, K)``.

Bootstrap CIs resample users. ``olist.stats.effect_sizes.bootstrap_ci`` is a
two-sample helper, so a one-sample, vectorised version lives here.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from recsys.config import EVAL_KS, RANDOM_SEED

Recs = Mapping[int, Sequence[int]]
Truth = Mapping[int, set[int]]

USER_METRICS = ("recall", "ndcg", "map", "hit_rate")


def _hits(recs: Sequence[int], relevant: set[int], k: int) -> list[bool]:
    """Whether each of the first ``k`` positions is a first-time relevant hit."""
    seen: set[int] = set()
    out = []
    for item in list(recs)[:k]:
        hit = item in relevant and item not in seen
        seen.add(item)
        out.append(hit)
    return out


def recall_at_k(recs: Sequence[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(_hits(recs, relevant, k)) / len(relevant)


def hit_rate_at_k(recs: Sequence[int], relevant: set[int], k: int) -> float:
    return float(any(_hits(recs, relevant, k)))


def ndcg_at_k(recs: Sequence[int], relevant: set[int], k: int) -> float:
    """Binary-relevance NDCG; the ideal ranking puts min(|relevant|, k) hits first."""
    if not relevant:
        return 0.0
    dcg = sum(1.0 / np.log2(i + 2) for i, hit in enumerate(_hits(recs, relevant, k)) if hit)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return float(dcg / idcg)


def average_precision_at_k(recs: Sequence[int], relevant: set[int], k: int = 12) -> float:
    """H&M competition AP@K (normalised by min(|relevant|, K))."""
    if not relevant:
        return 0.0
    score, n_hits = 0.0, 0
    for i, hit in enumerate(_hits(recs, relevant, k)):
        if hit:
            n_hits += 1
            score += n_hits / (i + 1)
    return score / min(len(relevant), k)


def map_at_k(recs: Recs, truth: Truth, k: int = 12) -> float:
    """Mean AP@K over the users in ``truth``."""
    if not truth:
        return 0.0
    return float(
        np.mean([average_precision_at_k(recs.get(u, []), rel, k) for u, rel in truth.items()])
    )


def per_user_metrics(recs: Recs, truth: Truth, k: int) -> pd.DataFrame:
    """One row per ground-truth user with recall, ndcg, map and hit_rate at ``k``."""
    rows = []
    for u, rel in truth.items():
        r = recs.get(u, [])
        rows.append(
            (
                u,
                recall_at_k(r, rel, k),
                ndcg_at_k(r, rel, k),
                average_precision_at_k(r, rel, k),
                hit_rate_at_k(r, rel, k),
            )
        )
    return pd.DataFrame(rows, columns=["customer_id", *USER_METRICS]).set_index("customer_id")


def catalogue_coverage(recs: Recs, catalogue: Iterable[int], k: int) -> float:
    """Share of the catalogue that appears in at least one user's top ``k``."""
    cat = set(catalogue)
    if not cat:
        return 0.0
    shown = {item for r in recs.values() for item in list(r)[:k]}
    return len(shown & cat) / len(cat)


def popularity_percentiles(item_counts: pd.Series) -> pd.Series:
    """Percentile rank (0..1, 1 = most popular) of each article's purchase count."""
    return item_counts.rank(pct=True, method="average")


def mean_popularity_percentile(recs: Recs, item_counts: pd.Series, k: int) -> float:
    """Mean popularity percentile over all recommended slots (unknown articles count 0)."""
    pct = popularity_percentiles(item_counts).to_dict()
    vals = [pct.get(item, 0.0) for r in recs.values() for item in list(r)[:k]]
    return float(np.mean(vals)) if vals else 0.0


def gini(counts: np.ndarray) -> float:
    """Gini coefficient of non-negative counts (0 = equal exposure, ->1 = concentrated)."""
    x = np.sort(np.asarray(counts, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    i = np.arange(1, n + 1)
    return float(np.sum((2 * i - n - 1) * x) / (n * x.sum()))


def recommendation_gini(recs: Recs, catalogue: Iterable[int], k: int) -> float:
    """Gini of how often each catalogue article is recommended (zeros included)."""
    counts = dict.fromkeys(catalogue, 0)
    for r in recs.values():
        for item in list(r)[:k]:
            counts[item] = counts.get(item, 0) + 1
    return gini(np.fromiter(counts.values(), dtype=float, count=len(counts)))


def bootstrap_mean_ci(
    values: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = RANDOM_SEED,
    chunk: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Percentile bootstrap CI of the column means of ``values`` (users x metrics).

    All columns share the same resamples. Returns ``(low, high)`` arrays, one
    entry per column (scalars broadcast as 1-element arrays for a 1-D input).
    """
    v = np.asarray(values, dtype=float)
    if v.ndim == 1:
        v = v[:, None]
    n = len(v)
    if n == 0:
        nan = np.full(v.shape[1], np.nan)
        return nan, nan.copy()
    rng = np.random.default_rng(seed)
    means = np.empty((n_boot, v.shape[1]))
    for start in range(0, n_boot, chunk):
        size = min(chunk, n_boot - start)
        idx = rng.integers(0, n, (size, n))
        means[start : start + size] = v[idx].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2], axis=0)
    return low, high


def evaluate(
    recs: Recs,
    truth: Truth,
    ks: Sequence[int] = EVAL_KS,
    item_counts: pd.Series | None = None,
    catalogue: Iterable[int] | None = None,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Tidy table: one row per (metric, k) with value, bootstrap CI and user count.

    ``item_counts`` (purchases per article in the history the model was fit on)
    enables the popularity-bias rows; ``catalogue`` (default: the articles in
    ``item_counts``) enables coverage and the Gini of recommendation counts.
    Those catalogue-level rows have no CI.
    """
    if catalogue is None and item_counts is not None:
        catalogue = item_counts.index.tolist()
    catalogue = list(catalogue) if catalogue is not None else None
    rows = []
    for k in ks:
        per_user = per_user_metrics(recs, truth, k)
        low, high = bootstrap_mean_ci(per_user.to_numpy(), n_boot, alpha, seed)
        for j, metric in enumerate(USER_METRICS):
            rows.append((metric, k, float(per_user[metric].mean()), low[j], high[j], len(per_user)))
        if catalogue is not None:
            rows.append(
                ("coverage", k, catalogue_coverage(recs, catalogue, k), np.nan, np.nan, len(recs))
            )
            rows.append(
                ("rec_gini", k, recommendation_gini(recs, catalogue, k), np.nan, np.nan, len(recs))
            )
        if item_counts is not None:
            value = mean_popularity_percentile(recs, item_counts, k)
            rows.append(("popularity_pct", k, value, np.nan, np.nan, len(recs)))
    return pd.DataFrame(rows, columns=["metric", "k", "value", "ci_low", "ci_high", "n_users"])


def paired_bootstrap_diff(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = RANDOM_SEED,
) -> dict[str, float]:
    """Mean of ``b - a`` over the same users, with a paired percentile bootstrap CI.

    Resampling users (not the two systems separately) keeps the pairing, so the CI
    reflects the per-user difference rather than two independent spreads.
    """
    diff = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    low, high = bootstrap_mean_ci(diff, n_boot, alpha, seed)
    return {
        "diff": float(diff.mean()) if len(diff) else float("nan"),
        "ci_low": float(low[0]),
        "ci_high": float(high[0]),
        "n_users": len(diff),
    }

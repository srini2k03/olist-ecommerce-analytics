"""Two-group comparisons: tests, effect sizes and bootstrap intervals."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import stats


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Standardised mean difference (a - b) using the pooled standard deviation."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """P(a > b) - P(a < b), computed from the Mann-Whitney U statistic (O(n log n))."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    u = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
    return float(2 * u / (len(a) * len(b)) - 1)


def cliffs_label(delta: float) -> str:
    """Romano et al. (2006) thresholds."""
    d = abs(delta)
    if d < 0.147:
        return "negligible"
    if d < 0.33:
        return "small"
    if d < 0.474:
        return "medium"
    return "large"


def bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    statistic: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for a two-sample statistic (each group resampled)."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        vals[i] = statistic(rng.choice(a, len(a)), rng.choice(b, len(b)))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def mean_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(a) - np.mean(b))


@dataclass
class Comparison:
    n_a: int
    n_b: int
    mean_a: float
    mean_b: float
    mean_diff: float
    mean_diff_ci: tuple[float, float]
    welch_t: float
    welch_p: float
    mannwhitney_u: float
    mannwhitney_p: float
    cohens_d: float
    cohens_d_ci: tuple[float, float]
    cliffs_delta: float
    cliffs_delta_ci: tuple[float, float]
    cliffs_label: str


def compare_groups(
    a: np.ndarray, b: np.ndarray, n_boot: int = 2000, boot_size: int | None = 5000, seed: int = 0
) -> Comparison:
    """Full two-group comparison of ``a`` vs ``b``.

    Tests use all rows. Bootstrap CIs for d and Cliff's delta resample at most
    ``boot_size`` rows per group (they are slow statistics); the mean-difference
    CI uses the full groups.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    sa = rng.choice(a, min(len(a), boot_size), replace=False) if boot_size else a
    sb = rng.choice(b, min(len(b), boot_size), replace=False) if boot_size else b
    t = stats.ttest_ind(a, b, equal_var=False)
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    delta = cliffs_delta(a, b)
    return Comparison(
        n_a=len(a),
        n_b=len(b),
        mean_a=float(a.mean()),
        mean_b=float(b.mean()),
        mean_diff=mean_diff(a, b),
        mean_diff_ci=bootstrap_ci(a, b, mean_diff, n_boot, seed=seed),
        welch_t=float(t.statistic),
        welch_p=float(t.pvalue),
        mannwhitney_u=float(u.statistic),
        mannwhitney_p=float(u.pvalue),
        cohens_d=cohens_d(a, b),
        cohens_d_ci=bootstrap_ci(sa, sb, cohens_d, min(n_boot, 1000), seed=seed),
        cliffs_delta=delta,
        cliffs_delta_ci=bootstrap_ci(sa, sb, cliffs_delta, min(n_boot, 1000), seed=seed),
        cliffs_label=cliffs_label(delta),
    )

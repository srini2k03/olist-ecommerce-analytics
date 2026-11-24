"""Reusable statistics with no SciPy dependency (numpy + the standard library only).

The in-browser dashboard (Pyodide) imports this module, so it avoids SciPy; the heavier
tests that need SciPy live in ``olist.stats.inference``. ``tests/test_stats_core.py``
checks every function against SciPy/statsmodels reference implementations.

Contents: descriptive summaries with confidence intervals, Wilson intervals, the normal
two-proportion test, Cohen's h, percentile bootstrap, permutation test, multiple-testing
corrections (Bonferroni, Holm, Benjamini-Hochberg) and normal-approximation power and
sample size for proportions and means.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from statistics import NormalDist

import numpy as np

_N = NormalDist()


def norm_cdf(x: float) -> float:
    return _N.cdf(x)


def norm_ppf(p: float) -> float:
    return _N.inv_cdf(p)


def z_crit(alpha: float = 0.05, two_sided: bool = True) -> float:
    return norm_ppf(1 - alpha / 2) if two_sided else norm_ppf(1 - alpha)


# ---------------------------------------------------------------------------- descriptive
@dataclass(frozen=True)
class Summary:
    n: int
    mean: float
    sd: float
    se: float
    ci_low: float
    ci_high: float
    median: float
    p25: float
    p75: float
    minimum: float
    maximum: float

    def to_dict(self) -> dict:
        return asdict(self)


def describe(x, conf: float = 0.95) -> Summary:
    """Summary with a normal-approximation CI for the mean (fine for n in the hundreds+)."""
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    n = len(a)
    if n == 0:
        nan = math.nan
        return Summary(0, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan)
    sd = float(a.std(ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n)
    z = z_crit(1 - conf)
    q = np.quantile(a, [0.25, 0.5, 0.75])
    m = float(a.mean())
    return Summary(
        n, m, sd, se, m - z * se, m + z * se, float(q[1]), float(q[0]), float(q[2]),
        float(a.min()), float(a.max()),
    )  # fmt: skip


# ---------------------------------------------------------------------------- proportions
def wilson_interval(successes, n, conf: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval; vectorised. n = 0 gives (0, 1)."""
    k = np.asarray(successes, dtype=float)
    nn = np.asarray(n, dtype=float)
    z = z_crit(1 - conf)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(nn > 0, k / nn, 0.0)
        denom = 1 + z**2 / nn
        centre = (p + z**2 / (2 * nn)) / denom
        half = z * np.sqrt(p * (1 - p) / nn + z**2 / (4 * nn**2)) / denom
    low = np.where((nn > 0) & (k > 0), np.clip(centre - half, 0, 1), 0.0)
    high = np.where((nn > 0) & (k < nn), np.clip(centre + half, 0, 1), 1.0)
    return low, high


@dataclass(frozen=True)
class ProportionTest:
    p_a: float
    p_b: float
    diff: float
    diff_ci: tuple[float, float]
    rel_lift: float
    z: float
    p_value: float
    cohens_h: float

    def to_dict(self) -> dict:
        return asdict(self)


def cohens_h(p_a: float, p_b: float) -> float:
    """Effect size for two proportions: 2*asin(sqrt(p_b)) - 2*asin(sqrt(p_a))."""
    return 2 * math.asin(math.sqrt(p_b)) - 2 * math.asin(math.sqrt(p_a))


def two_proportion_test(
    x_a: int, n_a: int, x_b: int, n_b: int, alpha: float = 0.05
) -> ProportionTest:
    """Two-sided z-test with pooled SE; Wald CI (unpooled SE) for p_b - p_a."""
    p_a, p_b = x_a / n_a, x_b / n_b
    pooled = (x_a + x_b) / (n_a + n_b)
    se0 = math.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    z = (p_b - p_a) / se0 if se0 > 0 else 0.0
    p_value = 2 * (1 - norm_cdf(abs(z)))
    se = math.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    zc = z_crit(alpha)
    diff = p_b - p_a
    rel = diff / p_a if p_a > 0 else math.nan
    return ProportionTest(
        p_a, p_b, diff, (diff - zc * se, diff + zc * se), rel, z, p_value, cohens_h(p_a, p_b)
    )


# ---------------------------------------------------------------------------- resampling
def bootstrap_ci(
    data,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    conf: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for a one-sample statistic."""
    a = np.asarray(data, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    vals = np.array([statistic(a[i]) for i in idx])
    lo, hi = np.quantile(vals, [(1 - conf) / 2, 1 - (1 - conf) / 2])
    return float(lo), float(hi)


def bootstrap_diff_ci(
    a, b, statistic: Callable[[np.ndarray], float] = np.mean, n_boot: int = 2000,
    conf: float = 0.95, seed: int = 0,
) -> tuple[float, float]:  # fmt: skip
    """Percentile bootstrap CI for statistic(b) - statistic(a), resampling each group."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        vals[i] = statistic(b[rng.integers(0, len(b), len(b))]) - statistic(
            a[rng.integers(0, len(a), len(a))]
        )
    lo, hi = np.quantile(vals, [(1 - conf) / 2, 1 - (1 - conf) / 2])
    return float(lo), float(hi)


def permutation_test(a, b, n_perm: int = 5000, seed: int = 0) -> float:
    """Two-sided permutation p-value for the difference in means (b - a)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    observed = abs(b.mean() - a.mean())
    pooled = np.concatenate([a, b])
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        if abs(pooled[len(a) :].mean() - pooled[: len(a)].mean()) >= observed - 1e-12:
            count += 1
    return (count + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------- multiple testing
def bonferroni(pvals) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    return np.minimum(p * len(p), 1.0)


def holm(pvals) -> np.ndarray:
    """Holm step-down adjusted p-values (controls the family-wise error rate)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def benjamini_hochberg(pvals) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (controls the false discovery rate)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)[::-1]
    adj = np.empty(m)
    running = 1.0
    for k, i in enumerate(order):
        rank = m - k
        running = min(running, p[i] * m / rank)
        adj[i] = running
    return adj


# ---------------------------------------------------------------------------- power
def sample_size_proportions(
    p0: float, rel_mde: float, alpha: float = 0.05, power: float = 0.8
) -> int:
    """Per-arm n for a two-sided two-proportion z-test (unpooled variance)."""
    p1 = p0 * (1 + rel_mde)
    var = p0 * (1 - p0) + p1 * (1 - p1)
    return math.ceil((z_crit(alpha) + norm_ppf(power)) ** 2 * var / (p1 - p0) ** 2)


def power_proportions(p0: float, rel_mde: float, n_per_arm: int, alpha: float = 0.05) -> float:
    p1 = p0 * (1 + rel_mde)
    se = math.sqrt((p0 * (1 - p0) + p1 * (1 - p1)) / n_per_arm)
    return norm_cdf(abs(p1 - p0) / se - z_crit(alpha))


def mde_proportions(p0: float, n_per_arm: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest relative lift detectable with the given n (bisection on power)."""
    lo, hi = 1e-6, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if power_proportions(p0, mid, n_per_arm, alpha) < power:
            lo = mid
        else:
            hi = mid
    return hi


def sample_size_means(sd: float, mde_abs: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Per-arm n to detect an absolute difference in means with a two-sided z-test."""
    return math.ceil(2 * (z_crit(alpha) + norm_ppf(power)) ** 2 * sd**2 / mde_abs**2)


def power_means(sd: float, mde_abs: float, n_per_arm: int, alpha: float = 0.05) -> float:
    se = sd * math.sqrt(2 / n_per_arm)
    return norm_cdf(abs(mde_abs) / se - z_crit(alpha))

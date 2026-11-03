"""A/B-test design and analysis for conversion-style (binary) metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


def sample_size_per_arm(p0: float, rel_mde: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Per-arm n for a two-sided two-proportion z-test (unpooled variance formula)."""
    p1 = p0 * (1 + rel_mde)
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    var = p0 * (1 - p0) + p1 * (1 - p1)
    return int(np.ceil((z_a + z_b) ** 2 * var / (p1 - p0) ** 2))


def power_at_n(p0: float, rel_mde: float, n_per_arm: int, alpha: float = 0.05) -> float:
    p1 = p0 * (1 + rel_mde)
    se = np.sqrt((p0 * (1 - p0) + p1 * (1 - p1)) / n_per_arm)
    z_a = stats.norm.ppf(1 - alpha / 2)
    return float(stats.norm.cdf(abs(p1 - p0) / se - z_a))


def srm_check(n_a: int, n_b: int, expected_share_a: float = 0.5) -> tuple[float, float]:
    """Sample-ratio-mismatch chi-square test. Returns (chi2, p-value)."""
    total = n_a + n_b
    expected = [total * expected_share_a, total * (1 - expected_share_a)]
    res = stats.chisquare([n_a, n_b], expected)
    return float(res.statistic), float(res.pvalue)


@dataclass
class FrequentistResult:
    p_a: float
    p_b: float
    abs_diff: float
    abs_diff_ci: tuple[float, float]
    rel_lift: float
    z: float
    p_value: float


def two_proportion_test(x_a: int, n_a: int, x_b: int, n_b: int, alpha: float = 0.05):
    """Two-sided z-test (pooled SE) with a Wald CI (unpooled SE) for p_b - p_a."""
    p_a, p_b = x_a / n_a, x_b / n_b
    pooled = (x_a + x_b) / (n_a + n_b)
    se_pooled = np.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    z = (p_b - p_a) / se_pooled
    p_value = 2 * stats.norm.sf(abs(z))
    se = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    zc = stats.norm.ppf(1 - alpha / 2)
    diff = p_b - p_a
    return FrequentistResult(
        p_a, p_b, diff, (diff - zc * se, diff + zc * se), diff / p_a, float(z), float(p_value)
    )


@dataclass
class BayesianResult:
    prob_b_better: float
    expected_loss_choose_b: float
    expected_loss_choose_a: float
    lift_ci: tuple[float, float]


def beta_binomial(
    x_a: int,
    n_a: int,
    x_b: int,
    n_b: int,
    prior: tuple[float, float] = (1.0, 1.0),
    draws: int = 200_000,
    seed: int = 0,
) -> BayesianResult:
    """Beta-Binomial posterior comparison via Monte Carlo.

    Expected loss of choosing B = E[max(p_a - p_b, 0)], i.e. the conversion we
    expect to give up if B is in fact worse.
    """
    rng = np.random.default_rng(seed)
    a = rng.beta(prior[0] + x_a, prior[1] + n_a - x_a, draws)
    b = rng.beta(prior[0] + x_b, prior[1] + n_b - x_b, draws)
    lift = b / a - 1
    lo, hi = np.quantile(lift, [0.025, 0.975])
    return BayesianResult(
        prob_b_better=float((b > a).mean()),
        expected_loss_choose_b=float(np.maximum(a - b, 0).mean()),
        expected_loss_choose_a=float(np.maximum(b - a, 0).mean()),
        lift_ci=(float(lo), float(hi)),
    )


def peeking_false_positive_rate(
    p: float,
    n_per_arm: int,
    n_looks: int,
    n_sims: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Simulate A/A tests and compare stopping at the first significant look with a
    single fixed-horizon test. Returns both false-positive rates."""
    rng = np.random.default_rng(seed)
    looks = np.linspace(n_per_arm / n_looks, n_per_arm, n_looks).astype(int)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    # Cumulative conversions at each look for every simulation (vectorised).
    increments = np.diff(np.concatenate([[0], looks]))
    xa = np.cumsum(rng.binomial(increments, p, size=(n_sims, n_looks)), axis=1)
    xb = np.cumsum(rng.binomial(increments, p, size=(n_sims, n_looks)), axis=1)
    pa, pb = xa / looks, xb / looks
    pooled = (xa + xb) / (2 * looks)
    se = np.sqrt(pooled * (1 - pooled) * 2 / looks)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, (pb - pa) / se, 0.0)
    sig = np.abs(z) > z_crit
    return {
        "fixed_horizon_fpr": float(sig[:, -1].mean()),
        "peeking_fpr": float(sig.any(axis=1).mean()),
        "n_looks": n_looks,
        "n_sims": n_sims,
    }


def simulate_ab(
    p_control: float,
    true_rel_lift: float,
    n_total: int,
    share_a: float = 0.5,
    seed: int = 0,
) -> dict[str, int]:
    """Simulate a randomised experiment: each visitor is assigned at random and
    converts with the arm's true rate."""
    rng = np.random.default_rng(seed)
    in_a = rng.random(n_total) < share_a
    n_a = int(in_a.sum())
    n_b = n_total - n_a
    x_a = int(rng.binomial(n_a, p_control))
    x_b = int(rng.binomial(n_b, p_control * (1 + true_rel_lift)))
    return {"n_a": n_a, "x_a": x_a, "n_b": n_b, "x_b": x_b}

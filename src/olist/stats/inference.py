"""Hypothesis tests that need SciPy, and the stratified comparison used to re-verify the
late-delivery vs review-score association. Pipeline and notebooks only (not the browser).

Everything here describes associations in observational data. Late orders differ from
on-time ones in region, category, season and seller, so a gap in reviews is not the causal
effect of lateness; the stratified comparison only removes the confounding by the
stratifying variable.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats

from olist.stats.core import bootstrap_diff_ci, holm, z_crit
from olist.stats.effect_sizes import cliffs_delta, cliffs_label, cohens_d


@dataclass(frozen=True)
class TestResult:
    name: str
    statistic: float
    p_value: float

    def to_dict(self) -> dict:
        return asdict(self)


def welch_t(a, b) -> TestResult:
    r = stats.ttest_ind(np.asarray(b, float), np.asarray(a, float), equal_var=False)
    return TestResult("welch_t", float(r.statistic), float(r.pvalue))


def mann_whitney(a, b) -> TestResult:
    r = stats.mannwhitneyu(np.asarray(b, float), np.asarray(a, float), alternative="two-sided")
    return TestResult("mann_whitney_u", float(r.statistic), float(r.pvalue))


def chi_square(table) -> TestResult:
    r = stats.chi2_contingency(np.asarray(table, float), correction=False)
    return TestResult("chi_square", float(r.statistic), float(r.pvalue))


def compare_groups(a, b, n_boot: int = 2000, seed: int = 0) -> dict:
    """b vs a: means, difference with bootstrap CI, effect sizes and two tests."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    delta = cliffs_delta(b, a)
    return {
        "n_a": len(a),
        "n_b": len(b),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "diff": float(b.mean() - a.mean()),
        "diff_ci_bootstrap": bootstrap_diff_ci(a, b, n_boot=n_boot, seed=seed),
        "cohens_d": cohens_d(b, a),
        "cliffs_delta": delta,
        "cliffs_label": cliffs_label(delta),
        "welch": welch_t(a, b).to_dict(),
        "mann_whitney": mann_whitney(a, b).to_dict(),
    }


def stratified_mean_difference(
    df: pd.DataFrame,
    value: str,
    group: str,
    stratum: str,
    min_per_group: int = 30,
    alpha: float = 0.05,
) -> dict:
    """Mean of ``value`` for group 1 minus group 0 within each stratum, with Welch p-values,
    Holm-adjusted across strata, and an inverse-variance pooled (fixed-effect) difference.

    Strata where either group has fewer than ``min_per_group`` rows are skipped (reported).
    """
    rows = []
    for key, g in df.groupby(stratum, observed=True):
        x1 = g.loc[g[group] == 1, value].dropna().to_numpy(float)
        x0 = g.loc[g[group] == 0, value].dropna().to_numpy(float)
        if len(x1) < min_per_group or len(x0) < min_per_group:
            continue
        diff = x1.mean() - x0.mean()
        se = math.sqrt(x1.var(ddof=1) / len(x1) + x0.var(ddof=1) / len(x0))
        p = stats.ttest_ind(x1, x0, equal_var=False).pvalue
        rows.append(
            {"stratum": key, "n_1": len(x1), "n_0": len(x0), "mean_1": x1.mean(),
             "mean_0": x0.mean(), "diff": diff, "se": se, "p_value": float(p)}
        )  # fmt: skip
    out = pd.DataFrame(rows)
    if out.empty:
        return {"strata": [], "pooled_diff": math.nan, "pooled_ci": (math.nan, math.nan)}
    out["p_holm"] = holm(out["p_value"])
    z = z_crit(alpha)
    out["ci_low"] = out["diff"] - z * out["se"]
    out["ci_high"] = out["diff"] + z * out["se"]
    w = 1 / out["se"] ** 2
    pooled = float((w * out["diff"]).sum() / w.sum())
    pooled_se = float(math.sqrt(1 / w.sum()))
    return {
        "stratum": stratum,
        "strata_used": len(out),
        "strata_total": int(df[stratum].nunique()),
        "strata_significant_holm": int((out["p_holm"] < alpha).sum()),
        "strata_same_sign": int((np.sign(out["diff"]) == np.sign(pooled)).sum()),
        "pooled_diff": pooled,
        "pooled_ci": (pooled - z * pooled_se, pooled + z * pooled_se),
        "strata": out.round(6).to_dict(orient="records"),
    }


def late_vs_on_time(orders: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    """Re-verify the headline finding on delivered, reviewed orders (order_facts frame).

    Returns the raw comparison (late minus on-time) and the comparison stratified by
    customer state and by product category.
    """
    d = orders[orders["delay_days"].notna() & orders["review_score"].notna()].copy()
    d["is_late"] = (d["delay_days"] > 0).astype(int)
    late = d.loc[d.is_late == 1, "review_score"]
    on_time = d.loc[d.is_late == 0, "review_score"]
    res = compare_groups(on_time, late, n_boot=n_boot, seed=seed)
    res["share_1_star_late"] = float((late <= 1).mean())
    res["share_1_star_on_time"] = float((on_time <= 1).mean())
    res["by_state"] = stratified_mean_difference(d, "review_score", "is_late", "customer_state")
    res["by_category"] = stratified_mean_difference(d, "review_score", "is_late", "category")
    res["interpretation"] = (
        "Association in observational data: late orders score lower within every stratum "
        "tested, but lateness is not randomised, so this is not a causal effect estimate."
    )
    return res

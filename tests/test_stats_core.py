"""olist.stats.core (no SciPy) against SciPy / statsmodels reference implementations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint, proportions_ztest

from olist.stats import abtest, core, inference


def test_normal_helpers():
    assert core.norm_cdf(1.96) == pytest.approx(stats.norm.cdf(1.96))
    assert core.norm_ppf(0.975) == pytest.approx(stats.norm.ppf(0.975))


def test_describe_matches_numpy():
    x = np.random.default_rng(1).normal(10, 2, 500)
    s = core.describe(x)
    assert s.n == 500 and s.mean == pytest.approx(x.mean()) and s.sd == pytest.approx(x.std(ddof=1))
    assert s.ci_low < s.mean < s.ci_high
    assert core.describe([]).n == 0


@pytest.mark.parametrize(("k", "n"), [(0, 10), (3, 10), (50, 400), (999, 1000)])
def test_wilson_matches_statsmodels(k, n):
    lo, hi = core.wilson_interval(k, n)
    ref = proportion_confint(k, n, method="wilson")
    assert (float(lo), float(hi)) == pytest.approx(ref)


def test_wilson_handles_zero_n():
    lo, hi = core.wilson_interval(np.array([0, 5]), np.array([0, 10]))
    assert lo[0] == 0 and hi[0] == 1 and 0 < lo[1] < 0.5 < hi[1] < 1


def test_two_proportion_test_matches_statsmodels_and_old_module():
    res = core.two_proportion_test(702, 7016, 798, 6977)
    z, p = proportions_ztest([798, 702], [6977, 7016])
    assert res.z == pytest.approx(z) and res.p_value == pytest.approx(p)
    old = abtest.two_proportion_test(702, 7016, 798, 6977)
    assert res.diff_ci == pytest.approx(old.abs_diff_ci)
    assert res.cohens_h == pytest.approx(
        2 * np.arcsin(np.sqrt(798 / 6977)) - 2 * np.arcsin(np.sqrt(702 / 7016))
    )


def test_bootstrap_ci_covers_the_mean():
    x = np.random.default_rng(2).exponential(3, 800)
    lo, hi = core.bootstrap_ci(x, n_boot=500)
    assert lo < x.mean() < hi
    lo2, hi2 = core.bootstrap_diff_ci(x, x + 1, n_boot=300)
    assert lo2 < 1 < hi2


def test_permutation_test_agrees_with_welch_on_clear_differences():
    rng = np.random.default_rng(3)
    a, b = rng.normal(0, 1, 200), rng.normal(0.5, 1, 200)
    assert core.permutation_test(a, b, n_perm=999) < 0.01
    assert core.permutation_test(a, rng.normal(0, 1, 200), n_perm=999) > 0.01


@pytest.mark.parametrize(
    ("ours", "method"),
    [(core.bonferroni, "bonferroni"), (core.holm, "holm"), (core.benjamini_hochberg, "fdr_bh")],
)
def test_multiple_testing_matches_statsmodels(ours, method):
    p = np.array([0.001, 0.01, 0.02, 0.04, 0.3, 0.049, 0.5])
    assert ours(p) == pytest.approx(multipletests(p, method=method)[1])


def test_power_and_sample_size_match_old_scipy_module():
    assert core.sample_size_proportions(0.099, 0.15) == abtest.sample_size_per_arm(0.099, 0.15)
    assert core.power_proportions(0.099, 0.15, 6763) == pytest.approx(
        abtest.power_at_n(0.099, 0.15, 6763)
    )
    n = core.sample_size_proportions(0.1, 0.2)
    assert core.power_proportions(0.1, 0.2, n) >= 0.8
    assert core.mde_proportions(0.1, n) == pytest.approx(0.2, rel=0.01)
    n_m = core.sample_size_means(sd=10, mde_abs=1)
    assert core.power_means(10, 1, n_m) == pytest.approx(0.8, abs=0.005)


def test_stratified_difference_recovers_a_constant_effect():
    rng = np.random.default_rng(4)
    rows = []
    for s, base in zip("ABCD", [4.5, 4.0, 3.5, 4.2], strict=True):
        for g in (0, 1):
            n = 400
            rows.append(pd.DataFrame({"s": s, "g": g, "y": rng.normal(base - 1.5 * g, 1, n)}))
    df = pd.concat(rows)
    res = inference.stratified_mean_difference(df, "y", "g", "s")
    assert res["strata_used"] == 4 and res["strata_significant_holm"] == 4
    assert res["pooled_diff"] == pytest.approx(-1.5, abs=0.1)
    assert res["pooled_ci"][0] < -1.5 < res["pooled_ci"][1]


def test_compare_groups_direction():
    rng = np.random.default_rng(5)
    res = inference.compare_groups(rng.normal(4.3, 1, 2000), rng.normal(2.3, 1, 300), n_boot=200)
    assert res["diff"] == pytest.approx(-2.0, abs=0.15)
    assert res["diff_ci_bootstrap"][0] < res["diff"] < res["diff_ci_bootstrap"][1]
    assert res["cliffs_delta"] < -0.5 and res["cliffs_label"] == "large"

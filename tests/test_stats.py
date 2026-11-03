import numpy as np
import pytest
from scipy import stats

from olist.stats.abtest import (
    beta_binomial,
    peeking_false_positive_rate,
    power_at_n,
    sample_size_per_arm,
    simulate_ab,
    srm_check,
    two_proportion_test,
)
from olist.stats.effect_sizes import (
    bootstrap_ci,
    cliffs_delta,
    cliffs_label,
    cohens_d,
    compare_groups,
    mean_diff,
)


def test_cliffs_delta_matches_brute_force():
    rng = np.random.default_rng(1)
    a, b = rng.integers(1, 6, 300), rng.integers(1, 6, 200)
    brute = np.mean(np.sign(a[:, None] - b[None, :]))
    assert cliffs_delta(a, b) == pytest.approx(brute)


def test_cliffs_delta_extremes_and_labels():
    assert cliffs_delta([5, 6, 7], [1, 2, 3]) == pytest.approx(1)
    assert cliffs_delta([1, 2, 3], [5, 6, 7]) == pytest.approx(-1)
    assert cliffs_label(0.1) == "negligible"
    assert cliffs_label(-0.5) == "large"


def test_cohens_d_known_value():
    a = np.array([2.0, 4.0, 6.0])
    b = np.array([1.0, 3.0, 5.0])
    assert cohens_d(a, b) == pytest.approx(0.5)


def test_bootstrap_ci_covers_true_difference():
    rng = np.random.default_rng(2)
    a, b = rng.normal(1, 1, 2000), rng.normal(0, 1, 2000)
    lo, hi = bootstrap_ci(a, b, mean_diff, n_boot=500)
    assert lo < mean_diff(a, b) < hi
    analytic_width = 2 * 1.96 * np.sqrt(a.var() / len(a) + b.var() / len(b))
    assert hi - lo == pytest.approx(analytic_width, rel=0.2)


def test_compare_groups_detects_difference():
    rng = np.random.default_rng(3)
    res = compare_groups(rng.normal(0, 1, 3000), rng.normal(0.5, 1, 3000), n_boot=200)
    assert res.welch_p < 1e-6 and res.mannwhitney_p < 1e-6
    assert res.cohens_d < 0 and res.cliffs_delta < 0
    assert res.cohens_d_ci[0] < res.cohens_d < res.cohens_d_ci[1]


def test_sample_size_gives_target_power():
    n = sample_size_per_arm(0.10, 0.10, power=0.8)
    assert 14_000 < n < 15_500  # textbook value for 10% -> 11%
    assert power_at_n(0.10, 0.10, n) == pytest.approx(0.8, abs=0.01)


def test_srm_check():
    assert srm_check(5000, 5000)[1] == pytest.approx(1.0)
    assert srm_check(5200, 4800)[1] < 0.001


def test_two_proportion_test_matches_chi_square():
    res = two_proportion_test(100, 1000, 130, 1000)
    chi2 = stats.chi2_contingency([[100, 900], [130, 870]], correction=False)
    assert res.p_value == pytest.approx(chi2.pvalue)
    assert res.abs_diff_ci[0] < res.abs_diff < res.abs_diff_ci[1]


def test_beta_binomial_symmetry_and_direction():
    same = beta_binomial(100, 1000, 100, 1000)
    assert same.prob_b_better == pytest.approx(0.5, abs=0.01)
    better = beta_binomial(100, 1000, 150, 1000)
    assert better.prob_b_better > 0.99
    assert better.expected_loss_choose_b < better.expected_loss_choose_a


def test_peeking_inflates_false_positives():
    res = peeking_false_positive_rate(0.1, 5000, n_looks=20, n_sims=1500)
    assert res["fixed_horizon_fpr"] == pytest.approx(0.05, abs=0.02)
    assert res["peeking_fpr"] > 0.15


def test_simulate_ab_is_reproducible():
    assert simulate_ab(0.1, 0.1, 10_000, seed=5) == simulate_ab(0.1, 0.1, 10_000, seed=5)

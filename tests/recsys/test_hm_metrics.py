"""Metric correctness on hand-computed examples, incl. the H&M MAP@12 edge cases."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from recsys.evaluate.metrics import (
    average_precision_at_k,
    bootstrap_mean_ci,
    catalogue_coverage,
    evaluate,
    gini,
    hit_rate_at_k,
    map_at_k,
    mean_popularity_percentile,
    ndcg_at_k,
    paired_bootstrap_diff,
    recall_at_k,
    recommendation_gini,
)


def reference_apk(actual, predicted, k=12):
    """The ml_metrics apk the H&M competition's MAP@12 is built on."""
    if len(predicted) > k:
        predicted = predicted[:k]
    score, num_hits = 0.0, 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    if not actual:
        return 0.0
    return score / min(len(actual), k)


def test_ap_hand_computed():
    assert average_precision_at_k([1, 2, 3, 4], {1, 3}, 12) == pytest.approx((1 + 2 / 3) / 2)
    assert average_precision_at_k([2, 1], {1}, 12) == pytest.approx(0.5)
    assert average_precision_at_k([5, 6, 7], {1}, 12) == 0.0


def test_ap_edge_cases():
    # Normalised by min(|relevant|, K): 12 hits out of 20 relevant is a perfect score.
    assert average_precision_at_k(list(range(12)), set(range(20)), 12) == pytest.approx(1.0)
    assert average_precision_at_k([], {1}, 12) == 0.0  # no predictions
    assert average_precision_at_k([1, 2], set(), 12) == 0.0  # nothing relevant
    # Only the first K predictions count.
    assert average_precision_at_k([0] * 12 + [1], {1}, 12) == 0.0
    # A repeated article uses a slot but scores once.
    assert average_precision_at_k([1, 1, 2], {1, 2}, 12) == pytest.approx((1 + 2 / 3) / 2)
    # Fewer than K predictions are not penalised.
    assert average_precision_at_k([1], {1}, 12) == 1.0


def test_ap_matches_the_competition_reference_on_random_cases():
    rng = np.random.default_rng(0)
    for _ in range(500):
        actual = set(rng.integers(0, 30, rng.integers(0, 15)).tolist())
        predicted = rng.integers(0, 30, rng.integers(0, 20)).tolist()
        assert average_precision_at_k(predicted, actual, 12) == pytest.approx(
            reference_apk(actual, predicted, 12)
        )


def test_map_averages_over_ground_truth_users_and_missing_users_score_zero():
    truth = {1: {10}, 2: {20}, 3: {30}}
    recs = {1: [10], 2: [99, 20], 4: [30]}
    assert map_at_k(recs, truth, 12) == pytest.approx((1 + 0.5 + 0) / 3)
    assert map_at_k(recs, {}, 12) == 0.0


def test_recall_ndcg_hit_rate_hand_computed():
    recs, rel = [5, 1, 7], {1, 2}
    assert recall_at_k(recs, rel, 3) == 0.5
    assert recall_at_k(recs, rel, 1) == 0.0
    assert hit_rate_at_k(recs, rel, 3) == 1.0 and hit_rate_at_k(recs, rel, 1) == 0.0
    expected = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(recs, rel, 3) == pytest.approx(expected)
    assert ndcg_at_k([1, 2], rel, 2) == pytest.approx(1.0)
    assert ndcg_at_k([1, 1], {1}, 2) == pytest.approx(1.0)  # duplicate does not inflate
    assert recall_at_k([1], set(), 3) == 0.0 and ndcg_at_k([1], set(), 3) == 0.0


def test_coverage_gini_and_popularity_bias():
    catalogue = list(range(10))
    recs = {1: [0, 1, 2], 2: [0, 1, 3]}
    assert catalogue_coverage(recs, catalogue, 2) == pytest.approx(0.2)
    assert catalogue_coverage(recs, catalogue, 3) == pytest.approx(0.4)
    assert gini(np.ones(5)) == pytest.approx(0.0)
    assert gini(np.array([0, 0, 0, 10])) == pytest.approx(0.75)
    assert gini(np.zeros(3)) == 0.0
    assert recommendation_gini({1: [0], 2: [0]}, range(4), 1) == pytest.approx(0.75)
    counts = pd.Series({0: 100, 1: 50, 2: 10, 3: 1})
    # percentiles: 0 -> 1.0, 1 -> 0.75, 2 -> 0.5, 3 -> 0.25; unknown article -> 0
    assert mean_popularity_percentile({1: [0, 1]}, counts, 2) == pytest.approx(0.875)
    assert mean_popularity_percentile({1: [3, 99]}, counts, 2) == pytest.approx(0.125)


def test_bootstrap_ci_shape_and_determinism():
    rng = np.random.default_rng(1)
    values = rng.random((400, 4))
    low, high = bootstrap_mean_ci(values, n_boot=300, seed=5)
    assert low.shape == high.shape == (4,)
    means = values.mean(axis=0)
    assert (low < means).all() and (means < high).all()
    low2, high2 = bootstrap_mean_ci(values, n_boot=300, seed=5)
    np.testing.assert_array_equal(low, low2)
    np.testing.assert_array_equal(high, high2)
    lo1, hi1 = bootstrap_mean_ci(np.full(50, 0.3), n_boot=100)
    assert lo1.shape == (1,) and lo1[0] == pytest.approx(0.3) and hi1[0] == pytest.approx(0.3)
    lo0, _ = bootstrap_mean_ci(np.empty((0, 2)))
    assert np.isnan(lo0).all()


def test_evaluate_returns_a_tidy_table():
    truth = {1: {10, 11}, 2: {12}, 3: {13}}
    recs = {1: [10, 12, 13], 2: [12, 10, 11], 3: [99, 98, 97]}
    counts = pd.Series({10: 5, 11: 4, 12: 3, 13: 2, 99: 1})
    table = evaluate(recs, truth, ks=(1, 3), item_counts=counts, n_boot=200)
    assert list(table.columns) == ["metric", "k", "value", "ci_low", "ci_high", "n_users"]
    assert len(table) == 2 * 7
    row = table.set_index(["metric", "k"])
    assert row.loc[("map", 1), "value"] == pytest.approx((1 + 1 + 0) / 3)
    assert row.loc[("recall", 3), "value"] == pytest.approx((0.5 + 1 + 0) / 3)
    assert row.loc[("hit_rate", 3), "value"] == pytest.approx(2 / 3)
    assert row.loc[("coverage", 1), "value"] == pytest.approx(3 / 5)
    user_rows = table[table.metric.isin(["recall", "ndcg", "map", "hit_rate"])]
    assert (user_rows.ci_low <= user_rows.value + 1e-12).all()
    assert (user_rows.value <= user_rows.ci_high + 1e-12).all()
    assert (user_rows.n_users == 3).all()
    assert table[table.metric == "coverage"].ci_low.isna().all()
    no_pop = evaluate(recs, truth, ks=(12,), n_boot=50)
    assert set(no_pop.metric) == {"recall", "ndcg", "map", "hit_rate"}


def test_paired_bootstrap_diff():
    a = np.array([0.0, 0.5, 1.0, 0.2] * 50)
    d = paired_bootstrap_diff(a, a + 0.1, n_boot=200)
    assert d["diff"] == pytest.approx(0.1)
    assert d["ci_low"] == pytest.approx(0.1) and d["ci_high"] == pytest.approx(0.1)
    rng = np.random.default_rng(3)
    b = a + rng.normal(0, 0.1, len(a))
    d2 = paired_bootstrap_diff(a, b, n_boot=300)
    assert d2["ci_low"] < d2["diff"] < d2["ci_high"] and d2["n_users"] == len(a)

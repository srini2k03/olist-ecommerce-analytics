"""Simulated A/B test built on olist.stats.abtest."""

from __future__ import annotations

import numpy as np
import pytest

from olist.stats.abtest import sample_size_per_arm
from recsys.abtest import decide, plan, run_ab_test, simulate


def _hits(p: float, n: int = 2000, seed: int = 0) -> np.ndarray:
    return (np.random.default_rng(seed).random(n) < p).astype(float)


def test_plan_sizes_for_the_shrunk_offline_lift():
    control = np.array([1, 0, 0, 0] * 250, dtype=float)  # 0.25
    treatment = np.array([1, 1, 0, 0] * 250, dtype=float)  # 0.50
    p = plan(control, treatment, shrinkage=0.5)
    assert p.p_control == 0.25 and p.offline_rel_lift == pytest.approx(1.0)
    assert p.expected_rel_lift == pytest.approx(0.5)
    assert p.n_per_arm == sample_size_per_arm(0.25, 0.5)
    assert plan(treatment, control).n_per_arm == 0  # no lift to test


def test_simulation_assigns_randomly_and_replays_outcomes():
    control = np.zeros(100)
    treatment = np.ones(100)
    c = simulate(control, treatment, 20_000, shrinkage=0.5, seed=1)
    assert c["n_a"] + c["n_b"] == 20_000
    assert abs(c["n_a"] / 20_000 - 0.5) < 0.02
    assert c["x_a"] == 0  # control users never convert here
    assert abs(c["x_b"] / c["n_b"] - 0.5) < 0.02  # half of the offline lift is realised


def test_decisions():
    control = _hits(0.30)
    better = np.maximum(control, _hits(0.15, seed=1))  # strictly more hits
    res = run_ab_test(control, better, "base", "model", seed=3)
    assert res["decision"] == "launch"
    assert res["srm"]["ok"] and res["lift_test"]["abs_diff_ci"][0] > 0
    assert run_ab_test(better, control, "a", "b")["decision"] == "no launch"
    res["srm"] = {"ok": False, "p_value": 1e-6}
    assert decide(res, 0.05)[0].startswith("no launch (investigate")
    res["srm"] = {"ok": True, "p_value": 0.5}
    res["lift_test"].update(p_value=0.4, abs_diff_ci=[-0.01, 0.02])
    assert decide(res, 0.05)[0] == "no launch (inconclusive)"


def test_launch_carries_a_caveat_when_offline_evidence_is_weak():
    control = _hits(0.30)
    better = np.maximum(control, _hits(0.15, seed=1))
    weak = {"diff": 0.01, "ci_low": -0.01, "ci_high": 0.03, "n_users": 400}
    res = run_ab_test(control, better, "base", "model", seed=3, offline_diff=weak)
    assert res["decision"] == "launch"
    assert any("Caveat" in r for r in res["reasoning"])

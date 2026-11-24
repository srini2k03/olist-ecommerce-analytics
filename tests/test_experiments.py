"""A/B simulator, decision engine, sensitivity and business scenarios."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from olist import scenarios
from olist.experiments import decision, simulator


@pytest.fixture(scope="module")
def frames(duck):
    return duck.query("SELECT * FROM fct_orders"), duck.query("SELECT * FROM fct_order_items")


@pytest.fixture(scope="module")
def base(frames):
    return simulator.Baseline.from_orders(*frames, period="2017-01-01/2018-09-01")


def test_baseline_is_calibrated_on_real_orders(frames, base):
    orders, _ = frames
    v = orders[orders.is_valid == 1]
    assert len(base.item_value) == len(v)
    assert base.multi_item_rate == pytest.approx((v.n_items >= 2).mean())
    assert 0 < base.qualifying_share(150) < 1


def test_plan_reaches_target_power(base):
    p = simulator.plan(base, simulator.Design(mde_rel=0.3))
    assert p["power_at_planned_n"] >= 0.79
    assert p["n_per_arm_planned"] >= p["n_per_arm_required"] * 0.97


def test_null_effect_has_no_lift_and_costs_the_subsidy(base):
    a = simulator.Assumptions(lift_multi_item=0.0, lift_orders=0.0)
    r = simulator.simulate(base, a, simulator.Design(duration_days=120, seed=3), n_boot=100)
    assert r["simulated"] is True
    assert abs(r["primary_multi_item_rate"]["diff"]) < 0.03
    econ = r["economics"]
    assert econ["subsidy_per_treatment_order"] > 0
    assert econ["incremental_contribution_per_month_at_rollout"] < 0  # pure cost
    assert decision.decide(r).verdict in {"do not ship", "inconclusive"}


def test_large_effect_is_detected(base):
    a = simulator.Assumptions(lift_multi_item=1.0, threshold=1e9)  # no subsidy cost
    r = simulator.simulate(base, a, simulator.Design(duration_days=200, seed=4), n_boot=100)
    prim = r["primary_multi_item_rate"]
    assert prim["p_value"] < 0.05 and prim["diff"] > 0 and prim["prob_b_better"] > 0.95
    assert r["economics"]["subsidy_per_treatment_order"] == 0
    assert r["secondary"]["items_per_order"]["diff"] > 0


def test_srm_breaks_the_decision():
    fake = {
        "srm_p_value": 1e-6,
        "primary_multi_item_rate": {
            "p_value": 0.001,
            "diff": 0.01,
            "p_a": 0.1,
            "p_b": 0.11,
            "prob_b_better": 0.99,
        },
        "guardrails": {"review_score": {"diff": 0.0, "diff_ci": (-0.01, 0.01)}},
        "economics": {
            "prob_contribution_positive": 0.99,
            "incremental_contribution_per_month_at_rollout": 1.0,
            "incremental_contribution_per_month_ci": (0.5, 1.5),
            "subsidy_per_treatment_order": 0.0,
        },
    }
    assert decision.decide(fake).verdict == "invalid experiment"
    fake["srm_p_value"] = 0.5
    assert decision.decide(fake).verdict == "ship"
    fake["guardrails"]["review_score"]["diff_ci"] = (-0.2, -0.1)
    assert decision.decide(fake).verdict == "do not ship"


def test_sensitivity_grid_is_monotonic_in_take_rate(base):
    a = simulator.Assumptions(lift_multi_item=0.15)
    g = simulator.sensitivity_grid(base, a, "take_rate", [0.05, 0.15], "lift_multi_item", [0.15])
    lo, hi = g.incremental_contribution_month.tolist()
    assert hi > lo  # a higher take rate makes the extra item value worth more
    t = simulator.tornado(base, a, {"take_rate": (0.05, 0.2), "threshold": (100.0, 300.0)})
    assert set(t.assumption) == {"take_rate", "threshold"} and (t.swing >= 0).all()


def test_expected_contribution_uses_common_random_numbers(base):
    a = simulator.Assumptions()
    assert simulator.expected_monthly_contribution(
        base, a
    ) == simulator.expected_monthly_contribution(base, a)
    b = replace(a, threshold=1e9)
    assert simulator.expected_monthly_contribution(
        base, b
    ) > simulator.expected_monthly_contribution(base, a)


def test_scenarios_label_inputs_and_move_in_the_right_direction(frames):
    orders, _ = frames
    b = scenarios.observed_baseline(orders)
    assert b.orders_per_month > 0 and 0 <= b.late_rate <= 1
    for name, fn in scenarios.SCENARIOS.items():
        res = fn(b)
        assert res["hypothetical"] is True and res["scenario"]
        assert {i["source"] for i in res["inputs"]} <= {"observed", "assumption"}
        assert any(i["source"] == "assumption" for i in res["inputs"]), name
        rows = pd.DataFrame(res["monthly"]).set_index("metric")
        assert np.isfinite(rows[["baseline", "scenario"]].to_numpy(float)).all(), name
    more = pd.DataFrame(scenarios.more_repeat_purchases(b, 0.02)["monthly"]).set_index("metric")
    assert more.loc["gmv", "change"] > 0
    sub = scenarios.shipping_subsidy(b, threshold=150, order_lift=0.0)
    rows = pd.DataFrame(sub["monthly"]).set_index("metric")
    assert rows.loc["contribution_estimate", "change"] < 0  # subsidy with no volume gain
    assert sub["subsidy_per_month"] > 0


def test_scenario_arithmetic_on_mini(mini):
    b = scenarios.observed_baseline(mini.query("SELECT * FROM fct_orders"))
    assert b.orders_per_month == pytest.approx(4 / 3)  # Jan, Feb, Mar 2017
    assert b.cancellation_rate == pytest.approx(2 / 6)
    res = scenarios.conversion_and_aov(b, order_change=0.10, aov_change=0.0, take_rate=0.10)
    rows = pd.DataFrame(res["monthly"]).set_index("metric")
    assert rows.loc["orders", "scenario"] == pytest.approx(4 / 3 * 1.1)
    assert rows.loc["marketplace_revenue_estimate", "baseline"] == pytest.approx(0.10 * 470 / 3)

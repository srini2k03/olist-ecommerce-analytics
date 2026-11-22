"""A simulated online A/B test for launching the recommender, built on ``olist.stats.abtest``.

Conversion is "the user buys at least one of the 12 recommended articles in the
week", whose offline estimate is HitRate@12. Two assumptions turn offline
numbers into an experiment, and both are reported:

* **Offline-to-online shrinkage.** Online lifts are usually smaller than
  offline ones (presentation, position bias, users who would have found the
  article anyway). The expected online lift is ``shrinkage x`` the offline
  relative lift; the power analysis is sized for it.
* **Replay outcomes.** Simulated users are drawn (with replacement) from the
  test-week users. A control user converts if the baseline's list would have
  hit; a treatment user converts if the new system's list would have hit, but
  only with probability ``shrinkage`` (otherwise they behave as in control), so
  the true simulated lift equals the expected online lift.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from olist.stats.abtest import (
    beta_binomial,
    power_at_n,
    sample_size_per_arm,
    srm_check,
    two_proportion_test,
)

MAX_USERS = 5_000_000


@dataclass
class ABPlan:
    p_control: float
    p_treatment_offline: float
    offline_rel_lift: float
    shrinkage: float
    expected_rel_lift: float
    alpha: float
    power: float
    n_per_arm: int


def plan(
    control_hits: np.ndarray,
    treatment_hits: np.ndarray,
    shrinkage: float = 0.5,
    alpha: float = 0.05,
    power: float = 0.8,
) -> ABPlan:
    p_c = float(np.mean(control_hits))
    p_t = float(np.mean(treatment_hits))
    offline = p_t / p_c - 1 if p_c > 0 else float("nan")
    expected = shrinkage * offline
    if not np.isfinite(expected) or expected <= 0 or p_c <= 0:
        n = 0
    else:
        n = sample_size_per_arm(p_c, expected, alpha, power)
    return ABPlan(p_c, p_t, offline, shrinkage, expected, alpha, power, n)


def simulate(
    control_hits: np.ndarray,
    treatment_hits: np.ndarray,
    n_total: int,
    shrinkage: float,
    seed: int,
    share_control: float = 0.5,
) -> dict[str, int]:
    """Randomly assign ``n_total`` resampled users and replay their outcomes."""
    rng = np.random.default_rng(seed)
    users = rng.integers(0, len(control_hits), n_total)
    in_control = rng.random(n_total) < share_control
    realised = rng.random(n_total) < shrinkage
    outcome = np.where(
        in_control | ~realised, np.asarray(control_hits)[users], np.asarray(treatment_hits)[users]
    )
    return {
        "n_a": int(in_control.sum()),
        "x_a": int(outcome[in_control].sum()),
        "n_b": int((~in_control).sum()),
        "x_b": int(outcome[~in_control].sum()),
    }


def run_ab_test(
    control_hits: np.ndarray,
    treatment_hits: np.ndarray,
    control_name: str,
    treatment_name: str,
    shrinkage: float = 0.5,
    alpha: float = 0.05,
    power: float = 0.8,
    seed: int = 42,
    srm_threshold: float = 0.001,
    offline_diff: dict | None = None,
) -> dict:
    """Plan, simulate and analyse the experiment; return the numbers and a decision."""
    p = plan(control_hits, treatment_hits, shrinkage, alpha, power)
    out: dict = {
        "control": control_name,
        "treatment": treatment_name,
        "plan": asdict(p),
        "offline_hit_rate_diff": offline_diff,
    }
    if p.n_per_arm == 0:
        out["decision"] = "no launch"
        out["reasoning"] = [
            f"Offline HitRate@12 does not favour {treatment_name} "
            f"({p.p_treatment_offline:.4f} vs {p.p_control:.4f}), so there is no lift to test."
        ]
        return out
    n_total = int(min(2 * p.n_per_arm, MAX_USERS))
    counts = simulate(control_hits, treatment_hits, n_total, shrinkage, seed)
    chi2, srm_p = srm_check(counts["n_a"], counts["n_b"])
    test = two_proportion_test(counts["x_a"], counts["n_a"], counts["x_b"], counts["n_b"], alpha)
    bayes = beta_binomial(counts["x_a"], counts["n_a"], counts["x_b"], counts["n_b"], seed=seed)
    achieved_power = power_at_n(
        p.p_control, p.expected_rel_lift, min(counts["n_a"], counts["n_b"]), alpha
    )
    out.update(
        {
            "experiment": {**counts, "n_total": n_total, "capped": n_total < 2 * p.n_per_arm},
            "srm": {"chi2": chi2, "p_value": srm_p, "ok": srm_p >= srm_threshold},
            "lift_test": {
                "p_control": test.p_a,
                "p_treatment": test.p_b,
                "abs_diff": test.abs_diff,
                "abs_diff_ci": list(test.abs_diff_ci),
                "rel_lift": test.rel_lift,
                "z": test.z,
                "p_value": test.p_value,
            },
            "bayes": {"prob_treatment_better": bayes.prob_b_better, "lift_ci": list(bayes.lift_ci)},
            "achieved_power": achieved_power,
        }
    )
    out["decision"], out["reasoning"] = decide(out, alpha)
    return out


def decide(res: dict, alpha: float) -> tuple[str, list[str]]:
    lt, srm, plan_ = res["lift_test"], res["srm"], res["plan"]
    reasons = [
        f"Offline HitRate@12 {plan_['p_control']:.4f} -> {plan_['p_treatment_offline']:.4f} "
        f"({plan_['offline_rel_lift']:+.1%}); planned for {plan_['shrinkage']:.0%} of it online "
        f"({plan_['expected_rel_lift']:+.1%}), {plan_['n_per_arm']:,} users per arm for "
        f"{plan_['power']:.0%} power at alpha {alpha}."
    ]
    if not srm["ok"]:
        reasons.append(
            f"Sample ratio mismatch (p = {srm['p_value']:.2g}): the result is not trustworthy."
        )
        return "no launch (investigate assignment)", reasons
    reasons.append(f"No sample ratio mismatch (p = {srm['p_value']:.2f}).")
    if res["experiment"]["capped"]:
        reasons.append(
            f"The simulated population was capped at {res['experiment']['n_total']:,} users, "
            f"so the achieved power is {res['achieved_power']:.0%}."
        )
    lo, hi = lt["abs_diff_ci"]
    reasons.append(
        f"Conversion {lt['p_control']:.4f} -> {lt['p_treatment']:.4f}, "
        f"{lt['abs_diff']:+.4f} (95% CI {lo:+.4f} to {hi:+.4f}), relative {lt['rel_lift']:+.1%}, "
        f"p = {lt['p_value']:.2g}."
    )
    if lt["p_value"] < alpha and lo > 0:
        reasons.append("The lift is positive and significant: launch, then keep monitoring.")
        off = res.get("offline_hit_rate_diff")
        if off and off["ci_low"] <= 0:
            reasons.append(
                f"Caveat: the offline paired HitRate@12 difference has a CI that includes zero "
                f"({off['ci_low']:+.4f} to {off['ci_high']:+.4f}, {off['n_users']:,} users). The "
                "simulation resamples those users as if they were the population, so it assumes "
                "the offline lift is real; a real experiment is what settles it."
            )
        return "launch", reasons
    if hi < 0:
        reasons.append("The treatment converts significantly worse: do not launch.")
        return "no launch", reasons
    reasons.append("The CI includes zero: not enough evidence of a lift, so do not launch yet.")
    return "no launch (inconclusive)", reasons

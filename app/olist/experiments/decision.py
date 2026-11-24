"""Decision engine for a (simulated) experiment: explicit, configurable rules.

Ship only if all hold: the experiment is valid (no sample-ratio mismatch), the primary
metric improved significantly, no guardrail got significantly worse beyond its tolerance,
and the incremental contribution is positive with enough probability. Otherwise "do not
ship" (a rule failed decisively) or "inconclusive" (not enough evidence either way).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DecisionRules:
    srm_alpha: float = 0.001
    alpha: float = 0.05
    guardrail_review_tolerance: float = 0.05  # stars: the CI lower bound may not go below -tol
    min_prob_contribution_positive: float = 0.80


@dataclass(frozen=True)
class Decision:
    verdict: str  # "ship" | "do not ship" | "inconclusive" | "invalid experiment"
    reasons: list[str]
    checks: dict[str, bool]
    rules: dict

    def to_dict(self) -> dict:
        return asdict(self)


def decide(result: dict, rules: DecisionRules = DecisionRules()) -> Decision:  # noqa: B008
    prim = result["primary_multi_item_rate"]
    review = result["guardrails"]["review_score"]
    econ = result["economics"]
    checks = {
        "no_sample_ratio_mismatch": result["srm_p_value"] >= rules.srm_alpha,
        "primary_metric_improved": prim["p_value"] < rules.alpha and prim["diff"] > 0,
        "guardrail_review_ok": review["diff_ci"][0] >= -rules.guardrail_review_tolerance,
        "contribution_positive": econ["prob_contribution_positive"]
        >= rules.min_prob_contribution_positive,
    }
    reasons = []
    if not checks["no_sample_ratio_mismatch"]:
        reasons.append(f"Sample-ratio mismatch (p = {result['srm_p_value']:.2g}): fix assignment.")
        return Decision("invalid experiment", reasons, checks, asdict(rules))
    reasons.append(
        f"Multi-item rate {prim['p_a']:.2%} -> {prim['p_b']:.2%} "
        f"(p = {prim['p_value']:.3g}, P(B>A) = {prim['prob_b_better']:.1%})."
    )
    reasons.append(
        f"Review guardrail: {review['diff']:+.3f} stars "
        f"(95% CI {review['diff_ci'][0]:+.3f} to {review['diff_ci'][1]:+.3f})."
    )
    lo, hi = econ["incremental_contribution_per_month_ci"]
    reasons.append(
        f"Incremental contribution at rollout: R$ "
        f"{econ['incremental_contribution_per_month_at_rollout']:,.0f}/month "
        f"(95% CI R$ {lo:,.0f} to R$ {hi:,.0f}; "
        f"P(> 0) = {econ['prob_contribution_positive']:.0%}); subsidy R$ "
        f"{econ['subsidy_per_treatment_order']:.2f} per treatment order."
    )
    if all(checks.values()):
        verdict = "ship"
    elif not checks["guardrail_review_ok"] or econ["prob_contribution_positive"] < 0.2:
        verdict = "do not ship"
    elif checks["primary_metric_improved"] and not checks["contribution_positive"]:
        verdict = "do not ship" if econ["prob_contribution_positive"] < 0.5 else "inconclusive"
    else:
        verdict = "inconclusive"
    return Decision(verdict, reasons, checks, asdict(rules))

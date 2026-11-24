"""Pages 9 and 10: the simulated A/B test with its decision engine, and business scenarios."""

from __future__ import annotations

import math
from dataclasses import replace

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from dashboard import charts
from dashboard.data import Filters, apply
from dashboard.ui import badge, download, empty, esc, header, palette, table
from olist import scenarios
from olist.experiments import decision, simulator


def _period(f: Filters) -> str:
    end = (pd.Timestamp(f"{f.end}-01") + pd.offsets.MonthBegin(1)).strftime("%Y-%m-%d")
    return f"{f.start}-01/{end}"


@st.cache_resource(show_spinner=False, max_entries=8)
def _baseline(f: Filters) -> simulator.Baseline:
    o, it = apply(f)
    return simulator.Baseline.from_orders(o, it, period=_period(f))


@st.cache_data(show_spinner="Simulating the experiment…", max_entries=16)
def _simulate(f: Filters, a: simulator.Assumptions, dsg: simulator.Design) -> dict:
    return simulator.simulate(_baseline(f), a, dsg, n_boot=300)


@st.cache_data(show_spinner="Running the sensitivity analysis…", max_entries=8)
def _sensitivity(f: Filters, a: simulator.Assumptions) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = _baseline(f)
    grid = simulator.sensitivity_grid(
        base, a, "lift_multi_item", [0.0, 0.1, 0.2, 0.3, 0.5], "take_rate", [0.05, 0.10, 0.15, 0.20]
    )
    return grid, simulator.tornado(base, a, simulator.DEFAULT_TORNADO)


# ---------------------------------------------------------------------------- 9 A/B testing
def ab_testing(ctx) -> None:
    header(
        "09 · A/B testing",
        "Simulated experiment + decision engine",
        "A/B testing",
        "Free shipping on orders whose item value reaches a threshold. The baskets are real "
        "orders from the selected period; the treatment effects are assumptions.",
    )
    badge(
        "simulated",
        "Olist never ran this test. Change the assumptions to see when the decision would flip.",
    )
    if empty(ctx.orders, min_rows=500):
        return
    base = _baseline(ctx.f)
    with st.container(border=True):
        st.markdown("**Assumptions and design**")
        c = st.columns(4)
        threshold = c[0].number_input(
            "Free shipping from (R$ item value)", 20.0, 1000.0, 150.0, 10.0, key="ab_threshold"
        )
        lift = c[1].slider(
            "Assumed lift in multi-item orders",
            0.0,
            1.0,
            0.15,
            0.05,
            format="%.2f",
            key="ab_lift",
            help="Relative change in the share of orders with 2+ items.",
        )
        lift_orders = c[2].slider(
            "Assumed change in order volume",
            -0.05,
            0.10,
            0.0,
            0.01,
            format="%.2f",
            key="ab_orders",
            help="Visits are not in the data, so conversion is an assumption.",
        )
        take = c[3].slider(
            "Take rate (assumed)", 0.05, 0.25, 0.10, 0.01, format="%.2f", key="ab_take"
        )
        c = st.columns(4)
        mde = c[0].slider(
            "Minimum detectable lift (relative)",
            0.05,
            0.50,
            0.15,
            0.05,
            format="%.2f",
            key="ab_mde",
        )
        alpha = c[1].select_slider("Significance level", [0.01, 0.05, 0.10], 0.05, key="ab_alpha")
        power = c[2].select_slider("Power", [0.7, 0.8, 0.9], 0.8, key="ab_power")
        pay = c[3].slider(
            "Payment cost (assumed)", 0.0, 0.04, 0.02, 0.005, format="%.3f", key="ab_pay"
        )
    a = simulator.Assumptions(
        threshold=threshold,
        lift_multi_item=lift,
        lift_orders=lift_orders,
        take_rate=take,
        payment_cost_rate=pay,
    )
    dsg = simulator.Design(alpha=alpha, power=power, mde_rel=mde)
    plan = simulator.plan(base, dsg)

    st.subheader("Plan")
    c = st.columns(5)
    c[0].metric(
        "Baseline multi-item rate",
        f"{plan['baseline_multi_item_rate']:.2%}",
        help="Observed share of valid orders with 2+ items in the selection.",
    )
    c[1].metric("Orders per day", f"{plan['daily_orders']:.0f}", help="Observed in the selection.")
    c[2].metric("Sample size per arm", f"{plan['n_per_arm_required']:,}")
    c[3].metric("Days needed (50/50)", f"{plan['days_required']}")
    c[4].metric(
        "Orders that already qualify",
        f"{base.qualifying_share(threshold):.1%}",
        help="They get free shipping without changing behaviour: pure subsidy.",
    )
    if plan["days_required"] > 365:
        st.warning(
            "The selection has too few orders to run this test in a year; widen the "
            "filters or accept a larger minimum detectable lift."
        )
        return

    res = _simulate(ctx.f, a, dsg)
    verdict = decision.decide(res)
    st.subheader(f"Decision engine: {verdict.verdict.upper()}")
    st.markdown("\n".join(f"- {esc(reason)}" for reason in verdict.reasons))
    checks = pd.DataFrame({"rule": list(verdict.checks), "passed": list(verdict.checks.values())})
    table(checks)

    prim, econ = res["primary_multi_item_rate"], res["economics"]
    st.subheader("Primary metric: share of multi-item orders")
    c = st.columns(5)
    c[0].metric(
        "Control", f"{prim['p_a']:.2%}", f"n = {res['n_a']:,}", delta_color="off", delta_arrow="off"
    )
    c[1].metric(
        "Treatment",
        f"{prim['p_b']:.2%}",
        f"n = {res['n_b']:,}",
        delta_color="off",
        delta_arrow="off",
    )
    c[2].metric(
        "Difference (pp)",
        f"{prim['diff'] * 100:+.2f}",
        f"95% CI {prim['diff_ci'][0] * 100:+.2f} to {prim['diff_ci'][1] * 100:+.2f}",
        delta_color="off",
        delta_arrow="off",
    )
    c[3].metric(
        "p-value",
        f"{prim['p_value']:.3g}",
        f"Cohen's h {prim['cohens_h']:.3f}",
        delta_color="off",
        delta_arrow="off",
    )
    c[4].metric(
        "P(treatment better)",
        f"{prim['prob_b_better']:.1%}",
        "Beta-Binomial posterior",
        delta_color="off",
        delta_arrow="off",
    )
    st.caption(
        f"SRM check p = {res['srm_p_value']:.3f} (assignment counts match the 50/50 "
        f"design). Simulated duration: {res['days']} days."
    )

    rows = []
    for name, r in {
        **res["secondary"],
        **res["guardrails"],
        "contribution_per_order": econ["incremental_contribution_per_order"],
    }.items():
        rows.append(
            {
                "metric": name,
                "control": r["mean_a"],
                "treatment": r["mean_b"],
                "difference": r["diff"],
                "ci_low": r["diff_ci_bootstrap"][0],
                "ci_high": r["diff_ci_bootstrap"][1],
                "p_value": r["p_value"],
                "cohens_d": r["cohens_d"],
            }
        )
    st.subheader("Secondary metrics, guardrail and economics (bootstrap 95% CIs)")
    sec = pd.DataFrame(rows)
    table(
        sec,
        {
            "control": "2",
            "treatment": "2",
            "difference": "3",
            "ci_low": "3",
            "ci_high": "3",
            "p_value": "3",
            "cohens_d": "3",
        },
    )
    c = st.columns(4)
    c[0].metric("Subsidy per treatment order", f"R$ {econ['subsidy_per_treatment_order']:.2f}")
    c[1].metric("Subsidy per month at rollout", f"R$ {econ['subsidy_per_month_at_rollout']:,.0f}")
    c[2].metric(
        "Contribution change per month*",
        f"R$ {econ['incremental_contribution_per_month_at_rollout']:,.0f}",
        f"P(> 0) = {econ['prob_contribution_positive']:.0%}",
        delta_color="off",
        delta_arrow="off",
    )
    c[3].metric(
        "Customer GMV change per month",
        f"R$ {econ['incremental_gmv_per_month_at_rollout']:,.0f}",
        help="Customers stop paying freight on qualifying orders, so GMV can fall.",
    )
    st.caption(
        "\\* Contribution = take rate x item value - payment cost - freight subsidy: an "
        "estimate from the assumptions above, not observed profit."
    )

    st.subheader("Sensitivity")
    grid, tor = _sensitivity(ctx.f, a)
    p = palette()
    c1, c2 = st.columns(2)
    with c1:
        heat = (
            alt.Chart(grid)
            .mark_rect()
            .encode(
                x=alt.X("lift_multi_item:O", title="Assumed lift in multi-item orders"),
                y=alt.Y("take_rate:O", title="Take rate", sort="descending"),
                color=alt.Color(
                    "incremental_contribution_month:Q",
                    title="R$/month",
                    scale=alt.Scale(scheme="redyellowgreen", domainMid=0),
                ),
                tooltip=[
                    "lift_multi_item",
                    "take_rate",
                    alt.Tooltip("incremental_contribution_month:Q", format=",.0f"),
                ],
            )
        )
        st.altair_chart(charts._cfg(heat, 260), width="stretch")
        st.caption("Expected contribution change per month at rollout (green = positive).")
    with c2:
        tl = tor.melt(["assumption", "centre"], ["at_low", "at_high"], var_name="end")
        bars = (
            alt.Chart(tl)
            .mark_bar(color=p["series2"])
            .encode(
                x=alt.X("min(value):Q", title="R$/month"),
                x2="max(value):Q",
                y=alt.Y("assumption:N", sort="-x", title=None),
                tooltip=[
                    "assumption",
                    alt.Tooltip("min(value):Q", format=",.0f"),
                    alt.Tooltip("max(value):Q", format=",.0f"),
                ],
            )
        )
        rule = alt.Chart(tor.head(1)).mark_rule(color=p["text"]).encode(x="centre:Q")
        st.altair_chart(charts._cfg(bars + rule, 260), width="stretch")
        st.caption("One assumption at a time between its low and high value (tornado).")
    download(sec, "ab_test_metrics", "Metrics table (CSV)")


# ---------------------------------------------------------------------------- 10 scenarios
PARAMS = {
    "Faster delivery": [("late_reduction", "Share of late deliveries fixed", 0.0, 1.0, 0.5, 0.05)],
    "Shipping subsidy": [
        ("threshold", "Free shipping from (R$ item value)", 50.0, 500.0, 150.0, 10.0),
        ("subsidy_share", "Share of freight subsidised", 0.0, 1.0, 1.0, 0.1),
        ("order_lift", "Assumed lift in orders", 0.0, 0.2, 0.03, 0.01),
    ],
    "More repeat purchases": [
        ("repeat_uplift_pp", "Extra share of new customers who return", 0.0, 0.10, 0.01, 0.005)
    ],
    "Fewer cancellations": [
        ("cancellation_reduction", "Share of cancellations avoided", 0.0, 1.0, 0.3, 0.05)
    ],
    "Better sellers": [
        ("percentile", "Fix sellers above this late-rate percentile", 50.0, 99.0, 90.0, 1.0)
    ],
    "Conversion and AOV": [
        ("order_change", "Change in orders (conversion)", -0.2, 0.3, 0.05, 0.01),
        ("aov_change", "Change in item value per order", -0.2, 0.3, 0.0, 0.01),
    ],
}


def business_simulation(ctx) -> None:
    header(
        "10 · Business simulation",
        "What-if scenarios",
        "Business simulation",
        "Each scenario starts from the observed baseline of the selection and applies "
        "explicit, labelled assumptions.",
    )
    badge(
        "hypothetical",
        "Outputs show what would follow if the assumptions held. They are "
        "not forecasts or measured effects.",
    )
    o = ctx.orders
    if empty(o, min_rows=200):
        return
    full = ctx.d["orders"]
    b = scenarios.observed_baseline(o, repeat_history=o)
    if not (math.isfinite(b.repeat_rate_after_on_time) and math.isfinite(b.repeat_rate_after_late)):
        b = replace(
            b,
            **dict(
                zip(
                    ["repeat_rate_after_on_time", "repeat_rate_after_late"],
                    scenarios.repeat_after_first_order(full),
                    strict=True,
                )
            ),
        )
        st.caption(
            "The selection is too short to measure repeat behaviour after a 180-day "
            "follow-up, so the repeat gap comes from the full data."
        )
    name = st.radio("Scenario", list(scenarios.SCENARIOS), horizontal=True, key="scen_name")
    with st.container(border=True):
        params = PARAMS[name]
        cols = st.columns(len(params) + 2)
        kwargs = {}
        for col, (key, label, lo, hi, default, step) in zip(cols, params, strict=False):
            kwargs[key] = col.slider(label, lo, hi, default, step, key=f"scen_{name}_{key}")
        kwargs["take_rate"] = cols[-2].slider(
            "Take rate (assumed)", 0.05, 0.25, 0.10, 0.01, key=f"scen_{name}_take"
        )
        kwargs["payment_cost_rate"] = cols[-1].slider(
            "Payment cost (assumed)", 0.0, 0.04, 0.02, 0.005, key=f"scen_{name}_pay"
        )
    res = scenarios.SCENARIOS[name](b, **kwargs)
    rows = pd.DataFrame(res["monthly"])
    rows["change_pct"] = rows["change"] / rows["baseline"].replace(0, np.nan)
    c = st.columns(4)
    r = rows.set_index("metric")
    c[0].metric(
        "Orders per month",
        f"{r.loc['orders', 'scenario']:,.0f}",
        f"{r.loc['orders', 'change']:+,.0f}",
    )
    c[1].metric(
        "GMV per month", f"R$ {r.loc['gmv', 'scenario']:,.0f}", f"R$ {r.loc['gmv', 'change']:+,.0f}"
    )
    c[2].metric(
        "Marketplace revenue* per month",
        f"R$ {r.loc['marketplace_revenue_estimate', 'scenario']:,.0f}",
        f"R$ {r.loc['marketplace_revenue_estimate', 'change']:+,.0f}",
    )
    c[3].metric(
        "Contribution* per month",
        f"R$ {r.loc['contribution_estimate', 'scenario']:,.0f}",
        f"R$ {r.loc['contribution_estimate', 'change']:+,.0f}",
    )
    extra = {
        k: v
        for k, v in res.items()
        if k in ("late_rate_after", "avg_review_change", "subsidy_per_month")
    }
    if extra:
        txt = []
        if "late_rate_after" in extra:
            txt.append(f"late rate {b.late_rate:.1%} → {extra['late_rate_after']:.1%}")
        if "avg_review_change" in extra:
            txt.append(f"mean review {extra['avg_review_change']:+.3f} stars (if causal)")
        if "subsidy_per_month" in extra:
            txt.append(f"freight subsidy R$ {extra['subsidy_per_month']:,.0f}/month")
        st.caption(esc("; ".join(txt).capitalize() + "."))
    st.markdown(f"**How to read it.** {res['notes']}")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Inputs**")
        inputs = pd.DataFrame(res["inputs"])[["label", "value", "source"]]
        table(inputs, {"value": "3"})
    with c2:
        st.markdown("**Monthly baseline vs scenario**")
        table(rows, {"baseline": "2", "scenario": "2", "change": "2", "change_pct": "%"})
    st.caption(
        "\\* Estimates: take rate and payment cost are assumptions (no commission or cost "
        "data in the dataset)."
    )
    download(rows.assign(scenario_name=name), "scenario")

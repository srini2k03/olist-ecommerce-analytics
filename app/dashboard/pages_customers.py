"""Pages 3, 4 and 5: customers (incl. CLV and the churn-risk lookup), cohort retention, RFM."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from dashboard import charts
from dashboard.data import read_optional
from dashboard.ui import badge, download, empty, header, kpis, palette, table
from olist.analytics import cohorts, rfm
from olist.metrics import assumption_values, compute_all


def _valid(o: pd.DataFrame) -> pd.DataFrame:
    return o[o["is_valid"] == 1]


# ---------------------------------------------------------------------------- 3 customers
def customers(ctx) -> None:
    d, o = ctx.d, ctx.orders
    header(
        "03 · Customers",
        "Acquisition, repeat purchase, value",
        "Customers",
        "A customer is a person (customer_unique_id); Olist issues a new customer_id for "
        "every order.",
    )
    if empty(o):
        return
    m = compute_all(o, ctx.items, assumption_values(d["metrics_spec"]))
    kpis(
        m,
        [
            "customers",
            "new_customers",
            "returning_customers",
            "repeat_rate",
            "gmv_per_customer",
            "orders_per_customer",
        ],
        d["metrics_spec"],
    )
    v = _valid(o)
    by_month = (
        pd.DataFrame(
            {
                "new": v[v["customer_order_seq"] == 1].groupby("purchase_month").size(),
                "returning": v[v["customer_order_seq"] > 1]
                .groupby("purchase_month")["customer_unique_id"]
                .nunique(),
            }
        )
        .fillna(0)
        .reset_index()
    )
    by_month["purchase_month"] = pd.to_datetime(by_month["purchase_month"])
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("New and returning customers by month")
        long = by_month.melt("purchase_month", ["new", "returning"], var_name="customers")
        st.altair_chart(
            charts.line(long, "purchase_month", "value", "Customers", color="customers"),
            width="stretch",
        )
    with c2:
        st.subheader("Time from first to second order")
        firsts = v[v["customer_order_seq"] == 1].set_index("customer_unique_id")["purchase_date"]
        seconds = v[v["customer_order_seq"] == 2].set_index("customer_unique_id")["purchase_date"]
        gap = (pd.to_datetime(seconds) - pd.to_datetime(firsts.reindex(seconds.index))).dt.days
        gap = gap.dropna()
        if len(gap):
            bins = [-1, 0, 7, 30, 90, 180, 365, 10_000]
            labels = ["same day", "1-7 d", "8-30 d", "31-90 d", "91-180 d", "181-365 d", "365+ d"]
            hist = pd.cut(gap, bins, labels=labels).value_counts().reindex(labels).reset_index()
            hist.columns = ["gap", "customers"]
            st.altair_chart(
                charts.bars(hist, "gap", "customers", horizontal=False, sort=labels, x_format=",d"),
                width="stretch",
            )
            st.caption(
                f"{(gap == 0).mean():.0%} of second orders are placed the same day as "
                "the first: split baskets, not loyalty."
            )
        else:
            st.info("No second orders in the selection.")

    st.subheader("Customer value concentration (within the selection)")
    per = v.groupby("customer_unique_id").agg(gmv=("gmv", "sum"), orders=("gmv", "size"))
    per = per.sort_values("gmv").reset_index()
    per["decile"] = rfm.ntile(len(per), 10) if len(per) >= 10 else 1
    dec = per.groupby("decile").agg(
        customers=("gmv", "size"),
        gmv=("gmv", "sum"),
        repeat_share=("orders", lambda s: float((s > 1).mean())),
    )
    dec["gmv_share"] = dec["gmv"] / dec["gmv"].sum()
    dec = dec.reset_index().sort_values("decile", ascending=False)
    table(dec, {"customers": "int", "gmv": "R$", "repeat_share": "%", "gmv_share": "%"})

    _clv(d)
    _churn_lookup(d)


def _clv(d: dict) -> None:
    st.subheader("Customer lifetime value: observed vs predicted")
    badge(
        "full",
        "BG/NBD + Gamma-Gamma fitted on the whole dataset as of 2018-09-01; the "
        "sidebar filters do not apply.",
    )
    clv = d.get("clv") or {}
    if not clv:
        st.info("CLV results are not in the bundle.")
        return
    obs, pred = clv["observed"], clv["predicted_12m"]
    hold = clv["holdout_validation"]
    c = st.columns(4)
    c[0].metric(
        "Observed GMV per customer",
        f"R$ {obs['observed_gmv_per_customer']:,.2f}",
        help="Historical GMV of valid orders before 2018-09-01.",
    )
    c[1].metric(
        "First order's share of observed GMV", f"{obs['first_order_share_of_observed_gmv']:.1%}"
    )
    c[2].metric(
        "Predicted next-12-month GMV per customer",
        f"R$ {pred['predicted_gmv_per_customer']:,.2f}",
        help="Expected repeat orders x expected GMV per repeat order (model estimate).",
    )
    c[3].metric(
        "Customers with > 0.1 expected repeat orders",
        f"{pred['customers_with_expected_repeat_above_0_1']:,}",
    )
    rows = [
        {
            "": "Actual",
            "repeat purchases": hold["actual"]["repeat_purchases"],
            "GMV (R$)": hold["actual"]["gmv"],
            "per-customer MAE (orders)": None,
            "ROC-AUC": None,
        },
        {
            "": "BG/NBD + Gamma-Gamma",
            "repeat purchases": hold["model"]["repeat_purchases"],
            "GMV (R$)": hold["model"]["gmv"],
            "per-customer MAE (orders)": hold["model"]["count_errors"]["mae"],
            "ROC-AUC": hold["model"]["ranking"]["roc_auc"],
        },
        {
            "": "Constant-rate baseline",
            "repeat purchases": hold["baseline_constant_rate"]["repeat_purchases"],
            "GMV (R$)": hold["baseline_constant_rate"]["gmv"],
            "per-customer MAE (orders)": hold["baseline_constant_rate"]["count_errors"]["mae"],
            "ROC-AUC": 0.5,
        },
    ]
    st.caption(
        f"Holdout check: fitted on purchases before {hold['calibration_end']}, predicting "
        f"{hold['customers']:,} existing customers' repeat purchases until "
        f"{hold['holdout_end']} ({hold['holdout_weeks']:.0f} weeks)."
    )
    table(
        pd.DataFrame(rows),
        {
            "repeat purchases": "1",
            "GMV (R$)": "R$",
            "per-customer MAE (orders)": "3",
            "ROC-AUC": "3",
        },
    )
    st.markdown(
        "Read this honestly: about 3% of customers ever return, so **almost all customer value "
        "is the first order**. The model is well calibrated **in aggregate** (useful for "
        "planning repeat revenue) but individual predictions are close to zero for nearly "
        "everyone; it ranks returners only modestly better than chance."
    )
    seg = read_optional("clv_by_segment.parquet")
    if seg is not None:
        table(
            seg[
                [
                    "segment",
                    "customers",
                    "observed_gmv_per_customer",
                    "predicted_gmv_12m_per_customer",
                    "predicted_share_of_total",
                ]
            ],
            {
                "customers": "int",
                "observed_gmv_per_customer": "R$",
                "predicted_gmv_12m_per_customer": "R$",
                "predicted_share_of_total": "%",
            },
        )
        st.caption(
            "Predicted share = predicted 12-month GMV / (observed + predicted). In BG/NBD a "
            "customer can only drop out right after a repeat purchase, so P(alive) is 100% for "
            "every one-time buyer (in the CSV); their low forecast comes from the low purchase "
            "rate, not from being 'alive'."
        )
        download(seg, "clv_by_segment")


def _churn_lookup(d: dict) -> None:
    st.subheader("Churn-risk lookup")
    badge("full", "LightGBM repeat-purchase model from notebook 04, scored on the whole dataset.")
    scores = read_optional("churn_scores.parquet")
    card = d.get("model_card") or {}
    if scores is None or not card:
        st.info("The churn model scores are not in the bundle.")
        return
    tm = card["test_metrics"]
    st.caption(
        f"Who is likely to buy again in the next 180 days? Scored as of {card['scoring_date']} "
        f"on point-in-time features. Held-out test (cutoff {card['eval_test_cutoff']}, "
        f"{card['test_rows']:,} customers): ROC-AUC {tm['roc_auc']:.3f}, PR-AUC "
        f"{tm['pr_auc']:.3f} vs base rate {tm['base_rate']:.2%}; top 10% captures "
        f"{tm['capture_top10']:.1%} of repeat buyers. A weak signal: use it to prioritise cheap "
        "actions."
    )
    examples = {
        "Most likely to return": scores.nlargest(1, "p_repeat_180d").customer_unique_id.iloc[0],
        "Median customer": scores.sort_values("p_repeat_180d")
        .iloc[len(scores) // 2]
        .customer_unique_id,
        "Least likely to return": scores.nsmallest(1, "p_repeat_180d").customer_unique_id.iloc[0],
    }
    ex = st.radio("Example customers", list(examples), horizontal=True, key="churn_example")
    cid = st.text_input("customer_unique_id", value=examples[ex], key="churn_id").strip().lower()
    row = scores[scores.customer_unique_id == cid]
    if row.empty:
        st.error("Customer id not found. Use a customer_unique_id from the Olist customers table.")
        return
    r = row.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "P(repeat in 180 days)",
        f"{r.p_repeat_180d:.2%}",
        f"{r.p_repeat_180d / tm['base_rate']:.1f}x base rate",
        delta_color="off",
        delta_arrow="off",
    )
    c2.metric("Churn risk", f"{1 - r.p_repeat_180d:.1%}")
    c3.metric(
        "Repeat decile",
        f"{int(r.repeat_decile)} of 10",
        "1 = most likely to return",
        delta_color="off",
        delta_arrow="off",
    )
    c4.metric("Segment (K-Means)", r.segment)
    st.markdown(
        f"**Profile:** {int(r.orders)} order(s), R\\$ {r.total_spend:,.2f} spent, last order "
        f"{int(r.recency_days)} days before {card['scoring_date']}, state {r.customer_state}, "
        f"last category `{r.last_category}`. **Why (top SHAP contributions to churn risk):**"
    )
    st.markdown("\n".join(f"- {x}" for x in r.reasons.split("; ")))


# ---------------------------------------------------------------------------- 4 cohorts
def cohort_matrix(ctx) -> tuple[pd.DataFrame, str]:
    v = _valid(ctx.orders)
    if ctx.f.item_filter:
        first = v.groupby("customer_unique_id")["purchase_month"].transform("min")
        v = v.assign(cohort_month=first)
        note = (
            "Item filters are active, so a cohort is the month of a customer's first order "
            "within the selected categories/sellers."
        )
    else:
        note = (
            "A cohort is the month of the customer's first order ever; only cohorts acquired "
            "inside the selected months are shown."
        )
    return cohorts.retention_matrix(v, max_months=18), note


def cohort_retention(ctx) -> None:
    header(
        "04 · Cohort retention",
        "Acquisition cohorts",
        "Cohort retention",
        "Share of each acquisition cohort that orders again k months later, and the GMV "
        "each cohort keeps generating.",
    )
    if empty(ctx.orders):
        return
    mat, note = cohort_matrix(ctx)
    st.caption(note)
    if mat.empty:
        st.info("No cohorts in the selection.")
        return
    curve = cohorts.retention_curve(mat, max_months=12)
    c = st.columns(4)
    m1 = curve.set_index("months_since_first")["retention"]
    c[0].metric("Cohorts", f"{mat.cohort_month.nunique()}")
    c[1].metric(
        "Month-1 retention",
        f"{m1.get(1, np.nan):.2%}" if 1 in m1 else "n/a",
        help="Size-weighted over cohorts observed for at least 1 month.",
    )
    c[2].metric("Month-3 retention", f"{m1.get(3, np.nan):.2%}" if 3 in m1 else "n/a")
    summ = cohorts.cohort_summary(mat)
    c[3].metric(
        "GMV after month 0",
        f"{summ['gmv_later_months'].sum() / summ['gmv_total'].sum():.1%}",
        help="Share of all cohort GMV that came after the acquisition month.",
    )

    view = st.radio(
        "Show",
        ["Retention (% of cohort)", "Cumulative GMV per customer (R$)"],
        horizontal=True,
        key="cohort_view",
    )
    heat = mat[mat.months_since_first <= 12].copy()
    heat["cohort"] = pd.to_datetime(heat["cohort_month"]).dt.strftime("%Y-%m")
    if view.startswith("Retention"):
        heat = heat[heat.months_since_first > 0]
        chart = charts.heatmap(
            heat,
            "months_since_first",
            "cohort",
            "retention",
            "% returning",
            tooltip=[
                "cohort",
                "months_since_first",
                alt.Tooltip("retention:Q", format=".2%"),
                alt.Tooltip("active_customers:Q", title="customers"),
                alt.Tooltip("cohort_size:Q", format=","),
            ],
        )
    else:
        chart = charts.heatmap(
            heat,
            "months_since_first",
            "cohort",
            "cumulative_gmv_per_customer",
            "R$ per customer",
            fmt=",.0f",
            tooltip=[
                "cohort",
                "months_since_first",
                alt.Tooltip("cumulative_gmv_per_customer:Q", format=",.2f"),
            ],
        )
    st.altair_chart(chart, width="stretch")

    st.subheader("Retention curve")
    p = palette()
    cv = curve[curve.months_since_first > 0]
    line = (
        alt.Chart(cv)
        .mark_line(point=True, color=p["series1"])
        .encode(
            x=alt.X("months_since_first:O", title="Months since first order"),
            y=alt.Y("retention:Q", title="% of cohort ordering", axis=alt.Axis(format=".1%")),
            tooltip=[
                "months_since_first",
                alt.Tooltip("retention:Q", format=".3%"),
                alt.Tooltip("cohorts:Q", title="cohorts observed"),
                alt.Tooltip("customers:Q", format=","),
            ],
        )
    )
    st.altair_chart(charts._cfg(line, 260), width="stretch")
    st.caption(
        "Each point averages only the cohorts observed for at least that many months "
        "(young cohorts cannot have month-6 activity yet)."
    )
    st.subheader("Cohort summary")
    summ["cohort_month"] = pd.to_datetime(summ["cohort_month"]).dt.strftime("%Y-%m")
    table(
        summ,
        {
            "cohort_size": "int",
            "gmv_month_0": "R$",
            "gmv_total": "R$",
            "gmv_later_months": "R$",
            "gmv_per_customer": "R$",
            "share_gmv_after_month_0": "%",
        },
    )
    c1, c2 = st.columns(2)
    with c1:
        download(mat, "cohort_matrix", "Cohort matrix (CSV)")
    with c2:
        download(summ, "cohort_summary", "Cohort summary (CSV)")


# ---------------------------------------------------------------------------- 5 RFM
def rfm_page(ctx) -> None:
    header(
        "05 · RFM segmentation",
        "Recency, frequency, monetary",
        "RFM segments",
        "Transparent rule-based segments on the filtered orders, scored as of the day after "
        "the selected period.",
    )
    o = ctx.orders
    if empty(o, min_rows=50):
        return
    as_of = pd.Timestamp(f"{ctx.f.end}-01") + pd.offsets.MonthBegin(1)
    scored = rfm.score(rfm.rfm_table(o, as_of))
    seg = rfm.segment_summary(scored)
    st.caption(
        f"Scored as of {as_of.date()} on {len(scored):,} customers. Recency and monetary "
        "are quintiles (NTILE, as in SQL 03); frequency is the order count because 97% of "
        "customers ordered once."
    )
    long = seg.melt("segment", ["customer_share", "revenue_share"], var_name="measure")
    long["measure"] = long["measure"].map(
        {"customer_share": "% of customers", "revenue_share": "% of revenue"}
    )
    st.altair_chart(
        charts.bars(
            long,
            "value",
            "segment",
            "",
            fmt=".1%",
            color="measure",
            x_format="%",
            sort=list(seg["segment"]),
            height=340,
        ),
        width="stretch",
    )
    table(
        seg[
            [
                "segment",
                "customers",
                "customer_share",
                "revenue_share",
                "aov",
                "avg_monetary",
                "repeat_rate",
                "avg_recency_days",
                "avg_review",
                "rule",
                "action",
            ]
        ],
        {
            "customers": "int",
            "customer_share": "%",
            "revenue_share": "%",
            "aov": "R$",
            "avg_monetary": "R$",
            "repeat_rate": "%",
            "avg_recency_days": "1",
            "avg_review": "2",
        },
    )
    download(seg, "rfm_segments")

    st.subheader("Drill down: one segment")
    pick = st.selectbox("Segment", list(seg["segment"]), key="rfm_seg")
    sample = scored.sample(min(3000, len(scored)), random_state=0)
    sample = sample.assign(highlight=sample["segment"] == pick)
    st.altair_chart(
        charts.scatter(
            sample,
            "recency_days",
            "monetary",
            "Days since last order",
            "Spend in the selection (R$, log)",
            highlight="highlight",
            log_y=True,
            tooltip=[
                "segment",
                "recency_days",
                "frequency",
                alt.Tooltip("monetary:Q", format=",.0f"),
            ],
        ),
        width="stretch",
    )
    members = scored.loc[scored["segment"] == pick, "customer_unique_id"]
    it = ctx.items[ctx.items["customer_unique_id"].isin(members) & (ctx.items["is_valid"] == 1)]
    top = (
        it.groupby("category", observed=True)["item_revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(8)
        .reset_index()
    )
    st.caption(
        f"Top categories bought by **{pick}** ({len(members):,} customers). "
        f"Suggested action: {rfm.SEGMENT_ACTIONS.get(pick, '')}"
    )
    table(top, {"item_revenue": "R$"})

    with st.expander("K-Means segments from notebook 03 (full data)"):
        km = read_optional("segments.parquet")
        if km is not None:
            st.caption(
                "k = 5 clusters on log-scaled RFM, named by explicit rules. The headline "
                "'at-risk big spenders' share comes from this clustering; the rule-based "
                "segments above are the transparent, filterable counterpart."
            )
            table(
                km[
                    [
                        "segment",
                        "customers",
                        "customer_share",
                        "revenue_share",
                        "recency_days",
                        "frequency",
                        "monetary",
                        "avg_review",
                        "action",
                    ]
                ],
                {
                    "customers": "int",
                    "customer_share": "%",
                    "revenue_share": "%",
                    "recency_days": "1",
                    "frequency": "2",
                    "monetary": "R$",
                    "avg_review": "2",
                },
            )

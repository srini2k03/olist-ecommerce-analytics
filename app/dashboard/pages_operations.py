"""Pages 6 and 7: sellers (uncertainty-aware ranking) and delivery (analysis + model card)."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from dashboard import charts
from dashboard.ui import badge, download, empty, header, kpis, palette, table
from olist.analytics import delivery, sellers
from olist.metrics import assumption_values, compute_all
from olist.stats.core import wilson_interval


# ---------------------------------------------------------------------------- 6 sellers
def sellers_page(ctx) -> None:
    d, o, it = ctx.d, ctx.orders, ctx.items
    header(
        "06 · Sellers",
        "Performance and fair ranking",
        "Sellers",
        "A ranking that is not revenue-only: service quality dominates, small samples are "
        "shrunk towards the average, and every rank comes with an interval.",
    )
    if empty(o):
        return
    m = compute_all(o, it, assumption_values(d["metrics_spec"]))
    card = sellers.scorecard_from_frames(o, it)
    c = st.columns([1, 1, 2])
    min_orders = c[0].number_input(
        "Minimum delivered orders to rank", 5, 500, sellers.MIN_ORDERS, 5, key="sel_min"
    )
    ranked = sellers.rank_sellers(card, min_orders=int(min_orders), n_draws=200)
    r = ranked[ranked["status"] == "ranked"]
    kpis(m, ["active_sellers", "item_revenue_per_seller"], d["metrics_spec"], per_row=2)
    c2 = st.columns(4)
    c2[0].metric(
        "Ranked sellers",
        f"{len(r):,}",
        f"{len(ranked) - len(r):,} below the minimum",
        delta_color="off",
        delta_arrow="off",
    )
    share = r["item_revenue"].sum() / max(ranked["item_revenue"].sum(), 1e-9)
    c2[1].metric("Item revenue covered by ranked sellers", f"{share:.0%}")
    if len(r):
        width = (r["rank_p95"] - r["rank_p05"]).median()
        c2[2].metric(
            "Median 90% rank interval",
            f"±{width / 2:.0f} places",
            help="Monte Carlo over each seller's Beta posteriors.",
        )
        top_rev = set(r.nlargest(25, "item_revenue")["seller_id"])
        top_dec = set(r[r["rank"] <= max(1, round(0.1 * len(r)))]["seller_id"])
        c2[3].metric(
            "Top-25 by revenue in the top decile",
            f"{len(top_rev & top_dec)} of 25",
            help="How much a revenue-only league would differ from this ranking.",
        )
    if r.empty:
        st.info("No seller reaches the minimum volume in this selection.")
        return
    labels = d["sellers"].set_index("code")
    r = r.assign(
        seller=r["seller_id"].map(lambda x: labels.loc[x, "seller_id"][:10] + "…"),
        state=r["seller_id"].map(labels["state"]),
    )

    st.subheader("Late rate vs volume")
    p_all = r["late_orders"].sum() / max(r["delivered_orders"].sum(), 1)
    volumes = np.geomspace(r["delivered_orders"].min(), r["delivered_orders"].max(), 60)
    grid = pd.DataFrame({"delivered_orders": np.unique(volumes.round())})
    lo, hi = wilson_interval(np.round(p_all * grid["delivered_orders"]), grid["delivered_orders"])
    grid["band_lo"], grid["band_hi"] = lo, hi
    st.altair_chart(
        charts.funnel(
            r,
            grid,
            "delivered_orders",
            "late_rate",
            "band_lo",
            "band_hi",
            p_all,
            tooltip=[
                "seller",
                "state",
                "delivered_orders",
                alt.Tooltip("late_rate:Q", format=".1%"),
                alt.Tooltip("late_rate_shrunk:Q", format=".1%"),
            ],
        ),
        width="stretch",
    )
    st.caption(
        "The shaded band is where a seller with the marketplace's average late rate would "
        "land by chance (Wilson 95%). Small sellers outside it are rarely meaningfully "
        "different, which is why the ranking uses shrunk rates."
    )

    st.subheader("Ranking")
    cols = [
        "rank",
        "rank_p05",
        "rank_p95",
        "p_top_decile",
        "seller",
        "state",
        "delivered_orders",
        "item_revenue",
        "late_rate",
        "late_rate_ci_low",
        "late_rate_ci_high",
        "late_rate_shrunk",
        "bad_review_rate_shrunk",
        "avg_review_shrunk",
        "cancellation_rate",
        "avg_delivery_days",
    ]
    table(
        r[cols].head(200),
        {
            "rank": "int",
            "rank_p05": "int",
            "rank_p95": "int",
            "p_top_decile": "%",
            "delivered_orders": "int",
            "item_revenue": "R$",
            "late_rate": "%",
            "late_rate_ci_low": "%",
            "late_rate_ci_high": "%",
            "late_rate_shrunk": "%",
            "bad_review_rate_shrunk": "%",
            "avg_review_shrunk": "2",
            "cancellation_rate": "%",
            "avg_delivery_days": "1",
        },
        height=420,
    )
    w = sellers.RANKING_WEIGHTS
    st.caption(
        "Score = "
        + " + ".join(f"{v:+.2f} x z({k})" for k, v in w.items())
        + " on shrunk rates (empirical-Bayes Beta priors fitted on ranked sellers). "
        "rank_p05-rank_p95: 90% interval; p_top_decile: probability of a top-10% rank."
    )
    download(r[cols].assign(seller_id=r["seller_id"].map(labels["seller_id"])), "seller_ranking")

    st.subheader("By seller state")
    st_tab = (
        card.assign(state=card["seller_id"].map(labels["state"]))
        .groupby("state")
        .agg(
            sellers=("seller_id", "size"),
            item_revenue=("item_revenue", "sum"),
            delivered=("delivered_orders", "sum"),
            late=("late_orders", "sum"),
        )
        .reset_index()
    )
    st_tab["late_rate"] = st_tab["late"] / st_tab["delivered"].where(st_tab["delivered"] > 0)
    table(
        st_tab.sort_values("item_revenue", ascending=False),
        {
            "sellers": "int",
            "item_revenue": "R$",
            "delivered": "int",
            "late": "int",
            "late_rate": "%",
        },
    )

    st.subheader("Drill down: one seller")
    options = r.head(200)["seller_id"].tolist()
    pick = st.selectbox(
        "Seller (top 200 by score)",
        options,
        key="sel_pick",
        format_func=lambda x: (
            f"#{int(r.set_index('seller_id').loc[x, 'rank'])} "
            f"{labels.loc[x, 'seller_id'][:10]}… ({labels.loc[x, 'state']})"
        ),
    )
    sub = it[(it["seller_id"] == pick) & (it["is_valid"] == 1)]
    orders_s = o[o.index.isin(sub["order_idx"])]
    mon = (
        orders_s.groupby("purchase_month")
        .agg(orders=("gmv", "size"), late_rate=("is_late", "mean"), review=("review_score", "mean"))
        .reset_index()
    )
    mon["purchase_month"] = pd.to_datetime(mon["purchase_month"])
    c1, c2 = st.columns(2)
    with c1:
        st.altair_chart(
            charts.line(mon, "purchase_month", "orders", "Orders per month"), width="stretch"
        )
    with c2:
        cats = (
            sub.groupby("category", observed=True)["item_revenue"]
            .sum()
            .sort_values(ascending=False)
            .head(6)
            .reset_index()
        )
        table(cats, {"item_revenue": "R$"})


# ---------------------------------------------------------------------------- 7 delivery
DELAY_BUCKETS = [-1000, -15, -8, 0, 3, 7, 14, 1000]
DELAY_LABELS = [
    "15+ days early",
    "8-14 early",
    "0-7 early",
    "1-3 late",
    "4-7 late",
    "8-14 late",
    "15+ late",
]


def delivery_page(ctx) -> None:
    d, o = ctx.d, ctx.orders
    header(
        "07 · Delivery",
        "Speed, promises, lateness",
        "Delivery",
        "Delivered orders only. Late = delivered after the date promised at checkout.",
    )
    if empty(o):
        return
    m = compute_all(o, ctx.items, assumption_values(d["metrics_spec"]))
    kpis(
        m,
        [
            "avg_delivery_days",
            "avg_promised_days",
            "late_rate",
            "avg_days_late_when_late",
            "avg_review",
            "review_gap_late",
        ],
        d["metrics_spec"],
    )
    dl = delivery.delivered(o)
    if dl.empty:
        st.info("No delivered orders in the selection.")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Promised vs actual days, by month")
        mon = (
            dl.groupby("purchase_month")
            .agg(
                promised=("estimated_days", "mean"),
                actual=("delivery_days", "mean"),
                late_rate=("is_late", "mean"),
            )
            .reset_index()
        )
        mon["purchase_month"] = pd.to_datetime(mon["purchase_month"])
        long = mon.melt("purchase_month", ["promised", "actual"], var_name="days")
        st.altair_chart(
            charts.line(long, "purchase_month", "value", "Days", color="days", y_format=",d"),
            width="stretch",
        )
    with c2:
        st.subheader("Late rate by month")
        st.altair_chart(
            charts.line(mon, "purchase_month", "late_rate", "Late rate", fmt=".1%", y_format="%"),
            width="stretch",
        )

    st.subheader("Review score by delay vs the promised date")
    rv = dl[dl["review_score"].notna()].copy()
    rv["bucket"] = pd.cut(rv["delay_days"], DELAY_BUCKETS, labels=DELAY_LABELS)
    b = (
        rv.groupby("bucket", observed=True)
        .agg(orders=("review_score", "size"), avg_review=("review_score", "mean"))
        .reset_index()
    )
    b["status"] = np.where(b["bucket"].astype(str).str.contains("late"), "late", "on time / early")
    st.altair_chart(
        charts.bars(
            b,
            "bucket",
            "avg_review",
            "Mean review (1-5)",
            fmt=".2f",
            horizontal=False,
            color="status",
            sort=DELAY_LABELS,
            x_format=".1f",
            height=280,
        ),
        width="stretch",
    )
    st.caption(
        "An association: late orders differ in region, category and season. The "
        "stratified analysis (late_vs_on_time.json) finds the gap within every state and "
        "category, but lateness was not randomised."
    )

    st.subheader("Late rate by customer state (95% Wilson intervals)")
    min_n = st.slider("Minimum delivered orders per state", 10, 1000, 100, 10, key="dl_min")
    bs = delivery.breakdown(o, "customer_state", min_orders=min_n).sort_values(
        "late_rate", ascending=False
    )
    st.altair_chart(
        charts.bars(
            bs,
            "late_rate",
            "customer_state",
            "Late rate",
            fmt=".1%",
            x_format="%",
            height=max(220, 18 * len(bs)),
            error=("late_rate_ci_low", "late_rate_ci_high"),
            tooltip=[
                "customer_state",
                alt.Tooltip("orders:Q", format=","),
                alt.Tooltip("late_rate:Q", format=".1%"),
                alt.Tooltip("avg_delivery_days:Q", format=".1f"),
                alt.Tooltip("avg_buffer_days:Q", format=".1f"),
            ],
        ),
        width="stretch",
    )

    c3, c4 = st.columns(2)
    lanes = dl.assign(
        lane=dl["seller_state"].astype(str) + " → " + dl["customer_state"].astype(str)
    )
    lt = delivery.breakdown(lanes, "lane")
    lt["share_of_late"] = lt["late_orders"] / max(lt["late_orders"].sum(), 1)
    with c3:
        st.subheader("Lanes with the most late orders")
        table(
            lt.sort_values("late_orders", ascending=False).head(12)[
                ["lane", "orders", "late_orders", "late_rate", "share_of_late"]
            ],
            {"orders": "int", "late_orders": "int", "late_rate": "%", "share_of_late": "%"},
        )
    with c4:
        st.subheader("By distance")
        dist = dl.assign(band=delivery.band(dl["distance_km"], delivery.DISTANCE_BANDS, " km"))
        bd = delivery.breakdown(dist, "band")
        table(
            bd[["band", "orders", "avg_delivery_days", "late_rate", "avg_review"]],
            {"orders": "int", "avg_delivery_days": "1", "late_rate": "%", "avg_review": "2"},
        )
    c5, c6 = st.columns(2)
    with c5:
        st.subheader("By freight charged")
        fr = dl.assign(band=delivery.band(dl["freight_value"], delivery.FREIGHT_BANDS, ""))
        bf = delivery.breakdown(fr, "band")
        bf["band"] = "R$ " + bf["band"].astype(str)
        table(
            bf[["band", "orders", "avg_delivery_days", "late_rate", "avg_review"]],
            {"orders": "int", "avg_delivery_days": "1", "late_rate": "%", "avg_review": "2"},
        )
    with c6:
        st.subheader("By category (main item)")
        bc = delivery.breakdown(o, "category", min_orders=200).sort_values(
            "late_rate", ascending=False
        )
        table(
            bc[["category", "orders", "avg_delivery_days", "late_rate", "avg_review"]].head(15),
            {"orders": "int", "avg_delivery_days": "1", "late_rate": "%", "avg_review": "2"},
        )
    c7, c8 = st.columns(2)
    with c7:
        download(bs, "delivery_by_state", "By state (CSV)")
    with c8:
        download(lt, "delivery_by_lane", "By lane (CSV)")
    _model_card(d.get("delivery_model") or {})


def _model_card(dm: dict) -> None:
    st.subheader("Can we predict delivery time and lateness at purchase?")
    badge(
        "full",
        "Trained and evaluated on the whole dataset with a time split; the filters do not apply.",
    )
    if not dm:
        st.info("The delivery-model results are not in the bundle.")
        return
    s = dm["splits"]
    st.caption(
        f"Train {s['train']['from']}..{s['train']['to_exclusive']} ({s['train']['orders']:,} "
        f"orders), validation {s['valid']['from']}..{s['valid']['to_exclusive']}, test "
        f"{s['test']['from']}..{s['test']['to_exclusive']} ({s['test']['orders']:,} orders). "
        f"Features are known at purchase time; seller and lane history only use earlier "
        f"deliveries. {dm['protocol'].capitalize()}."
    )
    reg = pd.DataFrame(dm["regression_delivery_days"]).T.reset_index(names="model")
    reg["mae_95ci"] = reg["mae_ci"].map(lambda x: f"{x[0]:.2f}-{x[1]:.2f}")
    clf = pd.DataFrame(
        {k: v for k, v in dm["classification_late"].items() if k != "lightgbm_regressor_vs_promise"}
    ).T.reset_index(names="model")
    clf["pr_auc_95ci"] = clf["pr_auc_ci"].map(lambda x: f"{x[0]:.3f}-{x[1]:.3f}")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Delivery days (test window)**")
        table(
            reg[["model", "mae", "mae_95ci", "rmse", "bias", "within_3_days"]],
            {"mae": "2", "rmse": "2", "bias": "2", "within_3_days": "%"},
        )
    with c2:
        st.markdown("**Late delivery (test window)**")
        table(
            clf[
                [
                    "model",
                    "pr_auc",
                    "pr_auc_95ci",
                    "roc_auc",
                    "recall_top10pct",
                    "brier",
                    "base_rate",
                ]
            ],
            {"pr_auc": "3", "roc_auc": "3", "recall_top10pct": "%", "brier": "3", "base_rate": "%"},
        )
    rob = dm.get("robustness_train_only", {})
    if rob:
        st.caption(
            "Robustness: fitted on 2017 only, the lane-history mean scores MAE "
            f"{rob['regression_mae']['lane_history_mean']:.2f} vs LightGBM "
            f"{rob['regression_mae']['lightgbm']:.2f} days, so the delivery-time gain depends on "
            "including the most recent months; for lateness LightGBM stays ahead (PR-AUC "
            f"{rob['classification_pr_auc']['lightgbm']:.3f} vs "
            f"{rob['classification_pr_auc']['lane_history_rate']:.3f})."
        )
    imp = pd.Series(dm["importance_gain_classifier"]).reset_index()
    imp.columns = ["feature", "gain_share"]
    p = palette()
    bar = (
        alt.Chart(imp)
        .mark_bar(color=p["series1"], cornerRadiusEnd=3)
        .encode(
            x=alt.X(
                "gain_share:Q",
                title="Share of total gain (late classifier)",
                axis=alt.Axis(format="%"),
            ),
            y=alt.Y("feature:N", sort="-x", title=None),
            tooltip=["feature", alt.Tooltip("gain_share:Q", format=".1%")],
        )
    )
    st.altair_chart(charts._cfg(bar, 300), width="stretch")
    st.markdown(
        "**Operational limits.** Olist's promise is padded by about two weeks, so it is a poor "
        "estimate of delivery time but rarely missed. The late-delivery model ranks risk about "
        "3x better than the base rate in its top decile; it is not calibrated after a regime "
        "change (the early-2018 surge), and the demand feature (orders in the previous 7 days) "
        "needs a live feed. Use it to prioritise proactive customer messages or carrier "
        "escalation, not to change promised dates without a test."
    )

"""Pages 1, 2 and 8: executive overview, revenue, products."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from dashboard import charts
from dashboard.ui import badge, download, empty, esc, header, kpis, palette, table
from olist.metrics import assumption_values, compute_all, format_value, metric_specs
from olist.stats.core import wilson_interval


def monthly(orders: pd.DataFrame, items: pd.DataFrame, item_filter: bool) -> pd.DataFrame:
    """Monthly orders and money; money from items when an item filter is active."""
    v = orders[orders["is_valid"] == 1]
    g = v.groupby("purchase_month", observed=True)
    out = pd.DataFrame(
        {
            "orders": g.size(),
            "customers": g["customer_unique_id"].nunique(),
            "payment_value": g["payment_value"].sum(),
        }
    )
    money = items[items["is_valid"] == 1] if item_filter else v
    mg = money.groupby("purchase_month", observed=True)
    out["item_revenue"] = mg["item_revenue"].sum()
    out["freight_value"] = mg["freight_value"].sum()
    out["gmv"] = out["item_revenue"] + out["freight_value"]
    out["aov"] = out["gmv"] / out["orders"]
    out["mom_growth"] = out["gmv"].pct_change()
    out = out.reset_index().rename(columns={"purchase_month": "month"})
    out["month"] = pd.to_datetime(out["month"])
    return out


def _metric_values(ctx) -> dict:
    return compute_all(ctx.orders, ctx.items, assumption_values(ctx.d["metrics_spec"]))


# ---------------------------------------------------------------------------- 1 overview
def overview(ctx) -> None:
    d, o = ctx.d, ctx.orders
    header(
        "01 · Executive overview",
        "Olist marketplace",
        "Olist marketplace analytics",
        "Brazilian e-commerce public dataset by Olist (CC BY-NC-SA 4.0). Every number on this "
        f"page follows the sidebar filters: {ctx.f.describe()}.",
    )
    if empty(o):
        return
    m = _metric_values(ctx)
    spec = d["metrics_spec"]
    kpis(
        m,
        [
            "gmv",
            "orders",
            "customers",
            "aov",
            "item_revenue",
            "freight_value",
            "payment_value",
            "marketplace_revenue_estimate",
            "late_rate",
            "avg_review",
            "repeat_rate",
            "cancellation_rate",
        ],
        spec,
    )
    take = assumption_values(spec)["take_rate"]
    st.caption(
        f"GMV = item revenue + freight. Payment value is what customers were charged. "
        f"\\* Marketplace revenue is an estimate: an assumed {take:.0%} take rate on item revenue "
        "(Olist's commissions are not in the data). Hover a card for its definition."
    )
    mon = monthly(o, ctx.items, ctx.f.item_filter)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("GMV by month")
        st.altair_chart(
            charts.line(
                mon,
                "month",
                "gmv",
                "GMV (R$)",
                tooltip=[
                    alt.Tooltip("month:T", format="%b %Y"),
                    alt.Tooltip("gmv:Q", format=",.0f"),
                    alt.Tooltip("mom_growth:Q", title="MoM", format="+.1%"),
                ],
            ),
            width="stretch",
        )
    with c2:
        st.subheader("Orders by month")
        pal = palette()
        bars = (
            alt.Chart(mon)
            .mark_bar(color=pal["series1"], cornerRadiusEnd=2)
            .encode(
                x=alt.X("yearmonth(month):T", title=None),
                y=alt.Y("orders:Q", title="Valid orders"),
                tooltip=[
                    alt.Tooltip("month:T", format="%b %Y"),
                    alt.Tooltip("orders:Q", format=","),
                    alt.Tooltip("customers:Q", format=","),
                    alt.Tooltip("aov:Q", title="AOV (R$)", format=",.2f"),
                ],
            )
        )
        st.altair_chart(charts._cfg(bars, 280), width="stretch")

    st.subheader("Key findings")
    badge(
        "full",
        "From the committed analysis of the whole dataset (2017-01 to 2018-08); "
        "the filters do not change these.",
    )
    _findings(d["findings"])

    st.subheader("Data quality")
    _data_quality(d["dq"])

    with st.expander("Money definitions: revenue, freight, GMV, payment value, estimates"):
        specs = metric_specs(spec)
        rows = [
            {
                "Metric": specs[n].label,
                "Kind": specs[n].kind,
                "Definition": specs[n].definition,
                "Value (filtered)": format_value(m[n], specs[n].unit),
            }
            for n in (
                "item_revenue",
                "freight_value",
                "gmv",
                "payment_value",
                "marketplace_revenue_estimate",
                "contribution_estimate",
            )
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    kpi_table = pd.DataFrame(
        [
            {"metric": n, "label": s.label, "unit": s.unit, "kind": s.kind, "value": m[n]}
            for n, s in metric_specs(spec).items()
        ]
    )
    download(kpi_table, "kpis", "Download all metrics (CSV)")


def _findings(f: dict) -> None:
    if not f:
        st.info("Analytics results are not in the bundle.")
        return
    lvo, rep, g = f["late_vs_on_time_review"], f["repeat"], f["growth"]
    lane, clv, ab = f["late_hotspot_lane"], f["clv_holdout"], f["ab_decision"]
    lines = [
        f"**Late deliveries and reviews:** late orders average **{lvo['late_mean']:.2f}★** "
        f"against **{lvo['on_time_mean']:.2f}★** on time (gap {lvo['diff']:+.2f}, 95% CI "
        f"{lvo['diff_ci'][0]:+.2f} to {lvo['diff_ci'][1]:+.2f}); lower in all "
        f"{lvo['states_tested']} states and {lvo['categories_tested']} categories tested. "
        "An association, not a measured causal effect.",
        f"**Customers rarely return:** {rep['repeat_rate']:.1%} of {rep['customers']:,} "
        f"customers ordered again; month-1 retention {rep['month1_retention']:.2%}; "
        f"{rep['same_day_second_orders']:.0%} of second orders are same-day split baskets.",
        f"**Growth has stalled:** Jan-Aug 2018 GMV was {g['gmv_jan_aug_2018_vs_2017']:+.0%} on "
        f"2017, but average month-on-month growth Feb-Aug 2018 was "
        f"{g['mean_mom_growth_feb_aug_2018']:+.1%}.",
        f"**Lateness is concentrated:** the {lane['lane'].replace('->', '→')} lane has "
        f"{lane['share_of_all_late']:.1%} of all late deliveries.",
        f"**CLV model is calibrated in aggregate:** BG/NBD predicted "
        f"{clv['predicted_repeat_purchases']:,.0f} repeat purchases for Mar-Aug 2018 vs "
        f"{clv['actual_repeat_purchases']:,} actual (a constant-rate baseline: "
        f"{clv['baseline_repeat_purchases']:,.0f}).",
        f"**Simulated free shipping from R$ 150:** decision engine says "
        f"**{ab['verdict']}** (the subsidy costs more than the extra basket value earns).",
    ]
    st.markdown(esc("\n".join(f"- {x}" for x in lines)))


def _data_quality(dq: dict) -> None:
    if not dq:
        st.info("The data-quality report is not in the bundle.")
        return
    s = dq["summary"]
    c = st.columns(4)
    c[0].metric("Checks", f"{s['checks']}")
    c[1].metric("Pass", f"{s['pass']}")
    c[2].metric("Warn (known, handled)", f"{s['warn']}")
    c[3].metric("Fail", f"{s['fail']}")
    rows = dq["row_counts"]
    st.caption(
        f"Layers: raw {sum(rows['raw'].values()):,} rows in {len(rows['raw'])} tables → "
        f"staging {len(rows['staging'])} models → marts {len(rows['marts'])} models "
        f"(fct_orders {rows['marts'].get('fct_orders', 0):,} rows). Status: "
        f"**{dq['status'].upper()}**: the pipeline stops on any failing check."
    )
    with st.expander("Checks that are not a clean pass"):
        nonpass = pd.DataFrame([c for c in dq["checks"] if c["status"] != "pass"])
        if nonpass.empty:
            st.write("All checks pass.")
        else:
            table(
                nonpass[["layer", "name", "status", "failing", "failing_share", "description"]],
                {"failing": "int", "failing_share": "%"},
            )
    with st.expander("All checks"):
        allc = pd.DataFrame(dq["checks"])[
            ["layer", "kind", "name", "severity", "status", "failing"]
        ]
        table(allc, {"failing": "int"}, height=360)


# ---------------------------------------------------------------------------- 2 revenue
def revenue(ctx) -> None:
    d, o, it = ctx.d, ctx.orders, ctx.items
    header(
        "02 · Revenue",
        "Item revenue, freight, GMV, payments",
        "Revenue",
        "Item revenue (prices), freight and payment value are kept apart; GMV is item "
        "revenue + freight.",
    )
    if empty(o):
        return
    m = _metric_values(ctx)
    kpis(
        m,
        ["item_revenue", "freight_value", "gmv", "payment_value", "aov", "gmv_mom_growth"],
        d["metrics_spec"],
    )
    mon = monthly(o, it, ctx.f.item_filter)
    st.subheader("Monthly item revenue and freight (stacked = GMV)")
    long = mon.melt("month", ["item_revenue", "freight_value"], var_name="component")
    p = palette()
    chart = (
        alt.Chart(long)
        .mark_bar()
        .encode(
            x=alt.X("yearmonth(month):T", title=None),
            y=alt.Y("value:Q", title="R$", stack=True, axis=alt.Axis(format="~s")),
            color=alt.Color(
                "component:N", title=None, scale=alt.Scale(range=[p["series1"], p["series2"]])
            ),
            tooltip=[
                alt.Tooltip("month:T", format="%b %Y"),
                "component",
                alt.Tooltip("value:Q", format=",.0f"),
            ],
        )
    )
    st.altair_chart(charts._cfg(chart, 300), width="stretch")

    c1, c2 = st.columns(2)
    v_items = it[it["is_valid"] == 1]
    by_cat = (
        v_items.groupby("category", observed=True)
        .agg(item_revenue=("item_revenue", "sum"), orders=("order_idx", "nunique"))
        .reset_index()
        .sort_values("item_revenue", ascending=False)
    )
    by_cat["share"] = by_cat["item_revenue"] / by_cat["item_revenue"].sum()
    with c1:
        st.subheader("Item revenue by category (top 15)")
        st.altair_chart(
            charts.bars(by_cat.head(15), "item_revenue", "category", "Item revenue (R$)"),
            width="stretch",
        )
    v = o[o["is_valid"] == 1]
    by_state = (
        v.groupby("customer_state", observed=True)
        .agg(gmv=("gmv", "sum"), orders=("gmv", "size"))
        .reset_index()
        .sort_values("gmv", ascending=False)
    )
    with c2:
        st.subheader("GMV by customer state (top 15)")
        st.altair_chart(
            charts.bars(by_state.head(15), "gmv", "customer_state", "GMV (R$)"), width="stretch"
        )

    st.subheader("Payment mix")
    pay = (
        v.groupby("main_payment_type", observed=True)
        .agg(
            orders=("gmv", "size"),
            payment_value=("payment_value", "sum"),
            avg_installments=("installments", "mean"),
        )
        .reset_index()
        .sort_values("orders", ascending=False)
    )
    pay["order_share"] = pay["orders"] / pay["orders"].sum()
    pay["value_share"] = pay["payment_value"] / pay["payment_value"].sum()
    table(
        pay,
        {
            "orders": "int",
            "payment_value": "R$",
            "avg_installments": "2",
            "order_share": "%",
            "value_share": "%",
        },
    )

    st.subheader("Payments vs GMV")
    rec = mon[["month", "gmv", "payment_value"]].copy()
    rec["payment_minus_gmv"] = rec["payment_value"] - rec["gmv"]
    st.caption(
        "Payment value includes instalment interest and vouchers and is recorded per order, "
        "so it is not split by item: under a category or seller filter it covers the whole "
        "matching orders."
    )
    table(rec, {"gmv": "R$", "payment_value": "R$", "payment_minus_gmv": "R$"})

    st.subheader("Drill down: one category over time")
    cats = by_cat["category"].astype(str).tolist()
    if cats:
        pick = st.selectbox("Category", cats, key="rev_cat")
        sub = v_items[v_items["category"] == pick]
        sm = (
            sub.groupby("purchase_month")
            .agg(item_revenue=("item_revenue", "sum"), orders=("order_idx", "nunique"))
            .reset_index()
        )
        sm["purchase_month"] = pd.to_datetime(sm["purchase_month"])
        st.altair_chart(
            charts.line(sm, "purchase_month", "item_revenue", f"{pick}: item revenue"),
            width="stretch",
        )
    c3, c4, c5 = st.columns(3)
    with c3:
        download(mon, "revenue_monthly", "Monthly table (CSV)")
    with c4:
        download(by_cat, "revenue_by_category", "By category (CSV)")
    with c5:
        download(by_state, "revenue_by_state", "By state (CSV)")


# ---------------------------------------------------------------------------- 8 products
def category_scorecard(orders: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    v = items[items["is_valid"] == 1]
    if v.empty:
        return pd.DataFrame()
    per_order = (
        v.groupby(["category", "order_idx"], observed=True)
        .agg(
            item_revenue=("item_revenue", "sum"),
            freight=("freight_value", "sum"),
            items=("item_revenue", "size"),
            review=("review_score", "first"),
            late=("is_late", "first"),
            customer=("customer_unique_id", "first"),
            month=("purchase_month", "first"),
        )
        .reset_index()
    )
    g = per_order.groupby("category", observed=True)
    out = pd.DataFrame(
        {
            "item_revenue": g["item_revenue"].sum(),
            "orders": g.size(),
            "items": g["items"].sum(),
            "buyers": g["customer"].nunique(),
            "avg_item_price": g["item_revenue"].sum() / g["items"].sum(),
            "freight_share_of_gmv": g["freight"].sum()
            / (g["item_revenue"].sum() + g["freight"].sum()),
            "avg_review": g["review"].mean(),
            "bad_review_rate": g["review"].apply(
                lambda s: float((s.dropna() <= 2).mean()) if s.notna().any() else np.nan
            ),
            "late_rate": g["late"].mean(),
        }
    )
    months = pd.to_datetime(per_order["month"])
    last = months.max()
    recent = per_order[months > last - pd.DateOffset(months=6)]
    prior = per_order[
        (months > last - pd.DateOffset(months=18)) & (months <= last - pd.DateOffset(months=12))
    ]
    r_now = recent.groupby("category", observed=True)["item_revenue"].sum()
    r_prev = prior.groupby("category", observed=True)["item_revenue"].sum()
    out["yoy_growth_6m"] = (r_now / r_prev.where(r_prev > 0) - 1).reindex(out.index)
    out["revenue_share"] = out["item_revenue"] / out["item_revenue"].sum()
    out = out.sort_values("item_revenue", ascending=False)
    out["cumulative_share"] = out["revenue_share"].cumsum()
    return out.reset_index()


def products(ctx) -> None:
    o, it = ctx.orders, ctx.items
    header(
        "08 · Products",
        "Category performance",
        "Products",
        "Revenue, popularity, satisfaction and delivery by product category (item-level "
        "attribution: an order spanning two categories counts in both).",
    )
    if empty(o):
        return
    sc = category_scorecard(o, it)
    if sc.empty:
        st.info("No items in the selection.")
        return
    c = st.columns(4)
    c[0].metric("Categories sold", f"{len(sc)}")
    top = sc.iloc[0]
    c[1].metric(
        "Top category",
        str(top["category"]),
        f"{top['revenue_share']:.1%} of item revenue",
        delta_color="off",
        delta_arrow="off",
    )
    core = int((sc["cumulative_share"] - sc["revenue_share"] < 0.8).sum())
    c[2].metric("Categories making 80% of revenue", f"{core}")
    c[3].metric("Median category review", f"{sc['avg_review'].median():.2f}★")

    st.subheader("Pareto: cumulative share of item revenue")
    sc["rank"] = np.arange(1, len(sc) + 1)
    p = palette()
    bar = (
        alt.Chart(sc.head(30))
        .mark_bar(color=p["series1"])
        .encode(
            x=alt.X("category:N", sort=None, title=None, axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("revenue_share:Q", title="Share of item revenue", axis=alt.Axis(format="%")),
            tooltip=[
                "category",
                alt.Tooltip("item_revenue:Q", format=",.0f"),
                alt.Tooltip("revenue_share:Q", format=".1%"),
                alt.Tooltip("cumulative_share:Q", format=".1%"),
            ],
        )
    )
    cum = (
        alt.Chart(sc.head(30))
        .mark_line(color=p["series2"], point=True)
        .encode(
            x=alt.X("category:N", sort=None),
            y=alt.Y("cumulative_share:Q", axis=alt.Axis(format="%")),
        )
    )
    st.altair_chart(
        charts._cfg(alt.layer(bar, cum).resolve_scale(y="independent"), 340), width="stretch"
    )

    st.subheader("Category scorecard")
    st.caption(
        "YoY growth compares the last 6 months of the selection with the same months a "
        "year earlier (blank when the selection is too short or the base is 0)."
    )
    min_orders = st.slider("Minimum orders", 0, 2000, 100, 50, key="prod_min")
    show = sc[sc["orders"] >= min_orders]
    table(
        show.drop(columns=["rank"]),
        {
            "item_revenue": "R$",
            "orders": "int",
            "items": "int",
            "buyers": "int",
            "avg_item_price": "R$",
            "freight_share_of_gmv": "%",
            "avg_review": "2",
            "bad_review_rate": "%",
            "late_rate": "%",
            "yoy_growth_6m": "%",
            "revenue_share": "%",
            "cumulative_share": "%",
        },
        height=420,
    )
    download(sc.drop(columns=["rank"]), "category_scorecard")

    st.subheader("Drill down: one category")
    pick = st.selectbox("Category", sc["category"].astype(str).tolist(), key="prod_cat")
    sub = it[(it["category"] == pick) & (it["is_valid"] == 1)]
    c1, c2 = st.columns(2)
    with c1:
        sm = sub.groupby("purchase_month").agg(item_revenue=("item_revenue", "sum")).reset_index()
        sm["purchase_month"] = pd.to_datetime(sm["purchase_month"])
        st.altair_chart(
            charts.line(sm, "purchase_month", "item_revenue", "Item revenue (R$)"), width="stretch"
        )
    with c2:
        top_s = (
            sub.groupby("seller_id")
            .agg(item_revenue=("item_revenue", "sum"), orders=("order_idx", "nunique"))
            .reset_index()
            .sort_values("item_revenue", ascending=False)
            .head(10)
        )
        sellers = ctx.d["sellers"].set_index("code")
        top_s["seller"] = top_s["seller_id"].map(lambda c: sellers.loc[c, "seller_id"][:10] + "…")
        top_s["state"] = top_s["seller_id"].map(sellers["state"])
        st.caption("Top sellers in the category")
        table(
            top_s[["seller", "state", "orders", "item_revenue"]],
            {"orders": "int", "item_revenue": "R$"},
        )
    dd = sub.drop_duplicates("order_idx")
    delivered = dd[dd["is_delivered"] == 1]
    if len(delivered):
        lo, hi = wilson_interval(delivered["is_late"].sum(), len(delivered))
        st.caption(
            f"{pick}: {len(delivered):,} delivered orders, mean delivery "
            f"{delivered['delivery_days'].mean():.1f} days, late rate "
            f"{delivered['is_late'].mean():.1%} (95% CI {float(lo):.1%} to {float(hi):.1%})."
        )

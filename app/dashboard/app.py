"""Dashboard shell: page config, sidebar filters, data-quality status and navigation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st

from dashboard import data
from dashboard.data import DEFAULT_END, DEFAULT_START, Filters
from dashboard.pages_business import overview, products, revenue
from dashboard.pages_customers import cohort_retention, customers, rfm_page
from dashboard.pages_experiments import ab_testing, business_simulation
from dashboard.pages_operations import delivery_page, sellers_page
from dashboard.ui import inject_css

TITLE = "Olist marketplace analytics"
PAGES = [
    # (title, url path, material icon, render function)
    ("Executive overview", "overview", ":material/dashboard:", overview),
    ("Revenue", "revenue", ":material/payments:", revenue),
    ("Customers", "customers", ":material/group:", customers),
    ("Cohort retention", "cohorts", ":material/grid_on:", cohort_retention),
    ("RFM segments", "rfm", ":material/donut_small:", rfm_page),
    ("Sellers", "sellers", ":material/storefront:", sellers_page),
    ("Delivery", "delivery", ":material/local_shipping:", delivery_page),
    ("Products", "products", ":material/category:", products),
    ("A/B testing", "ab-testing", ":material/science:", ab_testing),
    ("Business simulation", "simulation", ":material/tune:", business_simulation),
]
FILTER_KEYS = ("f_months", "f_cstate", "f_cat", "f_sstate", "f_seller")


@dataclass
class Context:
    d: dict
    f: Filters
    orders: pd.DataFrame
    items: pd.DataFrame


def _reset() -> None:
    for k in FILTER_KEYS:
        st.session_state.pop(k, None)


def sidebar(d: dict) -> Filters:
    sb = st.sidebar
    sb.markdown("### Filters")
    months = d["months"]
    default = (
        DEFAULT_START if DEFAULT_START in months else months[0],
        DEFAULT_END if DEFAULT_END in months else months[-1],
    )
    start, end = sb.select_slider(
        "Purchase months",
        months,
        value=default,
        key="f_months",
        help="2016 and Sep-Oct 2018 are partial months.",
    )
    o, it = d["orders"], d["items"]
    cstates = sb.multiselect(
        "Customer state",
        sorted(o["customer_state"].dropna().unique()),
        key="f_cstate",
        placeholder="All states",
    )
    cats = sb.multiselect(
        "Product category",
        sorted(it["category"].dropna().astype(str).unique()),
        key="f_cat",
        placeholder="All categories",
    )
    sstates = sb.multiselect(
        "Seller state",
        sorted(it["seller_state"].dropna().unique()),
        key="f_sstate",
        placeholder="All seller states",
    )
    top = it.groupby("seller_id")["item_revenue"].sum().sort_values(ascending=False).head(300)
    seller = sb.selectbox(
        "Seller (top 300 by revenue)",
        [None, *top.index.tolist()],
        key="f_seller",
        format_func=lambda c: "All sellers" if c is None else data.seller_label(d, c),
    )
    sb.button("Reset filters", on_click=_reset, icon=":material/restart_alt:")
    dq = d.get("dq") or {}
    if dq:
        s = dq["summary"]
        sb.markdown(
            f'<p class="sb-dq">Data quality: <b>{dq["status"].upper()}</b> · {s["checks"]} '
            f"checks · {s['pass']} pass · {s['warn']} warn · {s['fail']} fail</p>",
            unsafe_allow_html=True,
        )
    sb.caption(
        "Code: MIT. Data: Olist, CC BY-NC-SA 4.0. "
        "[Findings](https://srini2k03.github.io/olist-ecommerce-analytics/) · "
        "[Portfolio](https://srini2k03.github.io/)"
    )
    return Filters(
        start,
        end,
        tuple(cstates),
        tuple(cats),
        tuple(sstates),
        (int(seller),) if seller is not None else (),
    )


def main(only_page: str | None = None) -> None:
    """Run the dashboard; ``only_page`` renders one page without navigation (tests)."""
    st.set_page_config(page_title="Olist analytics", page_icon=":material/insights:", layout="wide")
    inject_css()
    d = data.load()
    f = sidebar(d)
    orders, items = data.apply(f)
    ctx = Context(d, f, orders, items)
    if only_page is not None:
        fn = next(p[3] for p in PAGES if p[0] == only_page)
        fn(ctx)
        return
    pages = [
        st.Page(lambda fn=fn: fn(ctx), title=title, url_path=path, icon=icon, default=(i == 0))
        for i, (title, path, icon, fn) in enumerate(PAGES)
    ]
    st.navigation(pages, position="sidebar").run()

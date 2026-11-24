"""Every query in sql/ is documented, runs, and passes sanity checks."""

from __future__ import annotations

import numpy as np
import pytest

from olist.sql_runner import DOMAINS, REQUIRED_HEADERS, frames_match, list_queries, run

QUERIES = list_queries()
BY_NAME = {q.name: q for q in QUERIES}


def test_queries_are_numbered_and_organised_by_domain():
    assert len(QUERIES) == 26
    assert [q.name[:2] for q in QUERIES] == [f"{i:02d}" for i in range(1, 27)]
    assert len({q.name for q in QUERIES}) == len(QUERIES)
    assert {q.domain for q in QUERIES} == set(DOMAINS)
    for domain in DOMAINS:
        assert sum(q.domain == domain for q in QUERIES) >= 2, domain


@pytest.mark.parametrize("query", QUERIES, ids=lambda q: q.name)
def test_analytics_read_marts_not_raw_tables(query):
    """Money and status logic lives in the marts; queries must not re-derive it."""
    import re

    body = "\n".join(ln for ln in query.sql.splitlines() if not ln.strip().startswith("--"))
    body = re.sub(r"EXTRACT\([^)]*\)", "", body)
    tables = set(re.findall(r"\b(?:FROM|JOIN)\s+([a-z_]+)", body))
    ctes = set(re.findall(r"\b([a-z_]+)\s+AS\s*\(", body))
    assert {t for t in tables - ctes if not t.startswith(("fct_", "dim_"))} == set()


@pytest.mark.parametrize("query", QUERIES, ids=lambda q: q.name)
def test_header_documents_question_method_output(query):
    text = "\n".join(line for line in query.sql.splitlines() if line.startswith("--"))
    for h in REQUIRED_HEADERS:
        assert h in text, f"{query.name} is missing '{h}'"
    assert all(len(v) > 20 for v in query.header.values())


def test_library_uses_joins_ctes_and_windows():
    sql = " ".join(q.sql.upper() for q in QUERIES)
    for token in ("JOIN", "WITH ", " OVER (", "LAG(", "LEAD(", "NTILE(", "PERCENTILE_CONT"):
        assert token in sql, token


@pytest.mark.parametrize("query", QUERIES, ids=lambda q: q.name)
def test_runs_on_duckdb(duck, query):
    df = run(duck, query)
    assert len(df) > 0
    assert len(df.columns) >= 3


def _q(engine, name):
    return run(engine, BY_NAME[name])


def test_cohort_month_zero_is_full_retention(duck):
    df = _q(duck, "02_cohort_retention")
    assert np.allclose(df.loc[df.months_since_first == 0, "retention"], 1.0)
    assert (df.retention <= 1).all()


def test_shares_sum_to_one(duck):
    assert _q(duck, "03_rfm_scores").customer_share.sum() == pytest.approx(1)
    assert _q(duck, "04_delivery_delay_vs_review").order_share.sum() == pytest.approx(1)
    assert _q(duck, "06_category_pareto").cumulative_share.iloc[-1] == pytest.approx(1)
    assert _q(duck, "11_basket_size").order_share.sum() == pytest.approx(1)
    assert _q(duck, "12_time_to_second_order").cumulative_share.iloc[-1] == pytest.approx(1)
    assert _q(duck, "15_customer_value_deciles").revenue_share.sum() == pytest.approx(1)
    assert _q(duck, "16_purchase_timing").order_share.sum() == pytest.approx(1)


def test_monthly_revenue_growth_is_consistent(duck):
    df = _q(duck, "01_monthly_revenue_mom")
    expected = df.gmv / df.gmv.shift(1) - 1
    assert np.allclose(df.mom_growth.iloc[1:], expected.iloc[1:])
    assert np.isnan(df.mom_growth.iloc[0])


def test_review_scores_and_rates_in_range(duck):
    df = _q(duck, "04_delivery_delay_vs_review")
    assert df.avg_review.between(1, 5).all()
    sla = _q(duck, "09_state_delivery_sla")
    assert sla.late_rate.between(0, 1).all()
    assert (sla.p90_delivery_days >= 0).all()


def test_repeat_rate_matches_value_deciles(duck):
    rep = _q(duck, "10_repeat_purchase_rate")
    overall = rep[rep.cohort_quarter == "All"].iloc[0]
    assert overall.repeat_customers <= overall.customers
    assert 0 < overall.repeat_rate < 1


def test_pareto_ranks_are_contiguous(duck):
    df = _q(duck, "06_category_pareto")
    assert df.revenue_rank.tolist() == list(range(1, len(df) + 1))
    assert df.revenue.is_monotonic_decreasing


@pytest.mark.postgres
@pytest.mark.parametrize("query", QUERIES, ids=lambda q: q.name)
def test_same_result_on_postgres(duck, pg, query):
    ok, msg = frames_match(run(duck, query), run(pg, query))
    assert ok, msg


def test_hotspot_share_is_of_all_late_orders(duck):
    """share_of_all_late must use every late order as the denominator, not only the late
    orders of lanes that pass the 200-order filter."""
    df = _q(duck, "13_late_delivery_hotspots")
    all_late = duck.query(
        """
        SELECT COUNT(*) AS n
        FROM orders AS o
        JOIN order_items AS oi ON oi.order_id = o.order_id AND oi.order_item_id = 1
        WHERE o.order_status = 'delivered'
          AND o.order_delivered_customer_date IS NOT NULL
          AND CAST(o.order_delivered_customer_date AS DATE)
              > CAST(o.order_estimated_delivery_date AS DATE)
        """
    ).n.iloc[0]
    assert np.allclose(df.share_of_all_late, df.late_orders / all_late)


def test_money_reconciliation_identities(duck):
    r = _q(duck, "18_money_reconciliation")
    assert np.allclose(r.gmv, r.item_revenue + r.freight_value)
    assert np.allclose(r.payment_minus_gmv, r.payment_value - r.gmv)
    assert r.freight_share_of_gmv.between(0, 1).all()


def test_order_funnel_partitions_orders(duck):
    f = _q(duck, "17_order_status_funnel")
    total = f.orders_cancelled + f.orders_delivered + f.orders_in_progress
    assert (total == f.orders_placed).all()
    assert f.cancellation_rate.between(0, 1).all()


def test_seller_scorecard_reconciles_with_items(duck):
    s = _q(duck, "21_seller_scorecard")
    revenue = duck.query("SELECT SUM(item_revenue) AS r FROM fct_order_items WHERE is_valid = 1")
    assert s.item_revenue.sum() == pytest.approx(revenue.r.iloc[0])
    assert (s.late_orders <= s.delivered_orders).all()
    assert (s.bad_reviews <= s.reviewed_orders).all()


def test_review_by_delivery_status_matches_buckets(duck):
    status = _q(duck, "25_review_by_delivery_status").set_index("delivery_status")
    buckets = _q(duck, "04_delivery_delay_vs_review")
    late = buckets[buckets.bucket_order >= 4]
    assert status.loc["late", "orders"] == late.orders.sum()
    assert status.loc["late", "avg_review"] < status.loc["on time", "avg_review"]

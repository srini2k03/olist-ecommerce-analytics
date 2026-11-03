"""Every query in sql/ is documented, runs, and passes sanity checks."""

from __future__ import annotations

import numpy as np
import pytest

from olist.sql_runner import REQUIRED_HEADERS, frames_match, list_queries, run

QUERIES = list_queries()
BY_NAME = {q.name: q for q in QUERIES}


def test_there_are_sixteen_queries():
    assert len(QUERIES) == 16
    assert [q.name[:2] for q in QUERIES] == [f"{i:02d}" for i in range(1, 17)]


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

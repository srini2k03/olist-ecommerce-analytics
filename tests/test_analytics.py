"""Cohorts, RFM, seller ranking, delivery breakdowns and CLV."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from olist.analytics import clv, cohorts, delivery, rfm, sellers
from olist.sql_runner import list_queries, run

BY_NAME = {q.name: q for q in list_queries()}


@pytest.fixture(scope="module")
def fo(duck):
    return duck.query("SELECT * FROM fct_orders")


def test_cohort_matrix_matches_sql(duck, fo):
    v = fo[(fo.is_valid == 1)]
    pd_ = pd.to_datetime(v.purchase_date)
    v = v[(pd_ >= "2017-01-01") & (pd_ < "2018-09-01")]
    v = v[pd.to_datetime(v.cohort_month) >= "2017-01-01"]
    m = cohorts.retention_matrix(v).sort_values(["cohort_month", "months_since_first"])
    sql = run(duck, BY_NAME["02_cohort_retention"])
    assert len(m) == len(sql)
    assert np.allclose(m.retention.to_numpy(), sql.retention.to_numpy())
    assert (m.cohort_size.to_numpy() == sql.cohort_size.to_numpy()).all()


def test_cohort_assignment_on_mini(mini):
    o = mini.query("SELECT * FROM fct_orders WHERE is_valid = 1")
    m = cohorts.retention_matrix(o).set_index(["cohort_month", "months_since_first"])
    jan = pd.Timestamp("2017-01-01")
    assert m.loc[(jan, 0), "cohort_size"] == 2
    assert m.loc[(jan, 2), "active_customers"] == 1  # u1 again in March
    assert m.loc[(jan, 2), "cumulative_gmv_per_customer"] == pytest.approx((165 + 88 + 220) / 2)


def test_retention_curve_only_uses_cohorts_old_enough():
    m = pd.DataFrame(
        {
            "cohort_month": pd.to_datetime(["2018-01-01", "2018-01-01", "2018-03-01"]),
            "months_since_first": [0, 2, 0],
            "active_customers": [100, 10, 50],
            "cohort_size": [100, 100, 50],
            "gmv": [1.0, 1.0, 1.0],
        }
    )
    curve = cohorts.retention_curve(m, last_month="2018-03-01").set_index("months_since_first")
    assert curve.loc[0, "customers"] == 150
    assert curve.loc[2, "cohorts"] == 1 and curve.loc[2, "retention"] == pytest.approx(0.1)
    assert 3 not in curve.index


def test_ntile_matches_sql_semantics():
    assert rfm.ntile(7, 5).tolist() == [1, 1, 2, 2, 3, 4, 5]
    assert rfm.ntile(10, 5).tolist() == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]


def test_rfm_segments_match_sql(duck, fo):
    ours = rfm.segment_summary(rfm.score(rfm.rfm_table(fo, "2018-09-01"))).set_index("segment")
    sql = run(duck, BY_NAME["03_rfm_scores"]).set_index("segment")
    assert set(ours.index) == set(sql.index)
    assert (ours.customers == sql.customers.reindex(ours.index)).all()
    assert np.allclose(ours.revenue_share, sql.revenue_share.reindex(ours.index))
    assert ours.customer_share.sum() == pytest.approx(
        1
    ) and ours.revenue_share.sum() == pytest.approx(1)
    assert (ours.loc[["Champions", "Loyal, lapsing"], "repeat_rate"] == 1).all()


def test_seller_scorecard_from_frames_matches_sql(duck):
    orders = duck.query("SELECT * FROM fct_orders")
    items = duck.query("SELECT * FROM fct_order_items")
    ours = sellers.scorecard_from_frames(orders, items).set_index("seller_id").sort_index()
    sql = run(duck, BY_NAME["21_seller_scorecard"]).set_index("seller_id").sort_index()
    for c in [
        "orders_placed",
        "orders",
        "item_revenue",
        "delivered_orders",
        "late_orders",
        "reviewed_orders",
        "bad_reviews",
        "cancelled_orders",
    ]:
        assert np.allclose(ours[c].astype(float), sql[c].astype(float)), c
    assert np.allclose(ours.avg_review, sql.avg_review, equal_nan=True)


def _card(delivered, late, revenue):
    n = len(delivered)
    return pd.DataFrame(
        {
            "seller_id": [f"s{i}" for i in range(n)],
            "orders_placed": delivered,
            "orders": delivered,
            "item_revenue": revenue,
            "delivered_orders": delivered,
            "late_orders": late,
            "reviewed_orders": delivered,
            "avg_review": [4.0] * n,
            "bad_reviews": [int(0.1 * d) for d in delivered],
            "cancelled_orders": [0] * n,
        }
    )


def test_ranking_needs_volume_and_shrinks_small_sellers():
    rng = np.random.default_rng(0)
    delivered = [40] * 30 + [2000, 35, 5]
    late = list(rng.binomial(40, 0.08, 30)) + [80, 0, 0]  # 4% late on 2,000; 0/35; 0/5
    card = _card(delivered, late, [1000.0] * 33)
    r = sellers.rank_sellers(card, n_draws=300).set_index("seller_id")
    assert r.loc["s32", "status"] == "insufficient volume" and np.isnan(r.loc["s32", "rank"])
    # the 0/35 seller is shrunk towards the average and is less certain than the big one
    assert r.loc["s31", "late_rate"] == 0 and r.loc["s31", "late_rate_shrunk"] > 0.02
    assert r.loc["s30", "late_rate_shrunk"] == pytest.approx(0.04, abs=0.005)
    width_small = r.loc["s31", "rank_p95"] - r.loc["s31", "rank_p05"]
    width_big = r.loc["s30", "rank_p95"] - r.loc["s30", "rank_p05"]
    assert width_small > width_big
    lo, hi = r.loc["s31", ["late_rate_ci_low", "late_rate_ci_high"]]
    assert lo == 0 and 0.05 < hi < 0.15  # Wilson interval for 0/35


def test_beta_prior_recovers_mean():
    rng = np.random.default_rng(1)
    p = rng.beta(8, 92, 500)
    n = rng.integers(50, 500, 500)
    a, b = sellers.beta_prior(rng.binomial(n, p), n)
    assert a / (a + b) == pytest.approx(0.08, abs=0.01)
    assert 30 < a + b < 300  # prior strength of the right order (true 100)


def test_delivery_breakdown_on_mini(mini):
    o = mini.query("SELECT * FROM fct_orders")
    b = delivery.breakdown(o, "customer_state").set_index("customer_state")
    assert b.loc["SP", "orders"] == 2 and b.loc["SP", "late_rate"] == 0.5
    assert b.loc["RJ", "avg_delivery_days"] == 12
    assert set(b.columns) >= {"late_rate_ci_low", "late_rate_ci_high", "avg_buffer_days"}


def _simulate_bgnbd(n, r, alpha, a, b, seed=0):
    rng = np.random.default_rng(seed)
    lam = rng.gamma(r, 1 / alpha, n)
    p = rng.beta(a, b, n)
    T = rng.uniform(30, 90, n)
    x = np.zeros(n)
    t_x = np.zeros(n)
    for i in range(n):
        t = 0.0
        while True:
            t += rng.exponential(1 / lam[i])
            if t > T[i]:
                break
            x[i] += 1
            t_x[i] = t
            if rng.random() < p[i]:
                break
    return x, t_x, T


def test_bgnbd_recovers_parameters_and_predicts_totals():
    x, t_x, T = _simulate_bgnbd(4000, r=0.3, alpha=10.0, a=1.5, b=5.0)
    m = clv.fit_bgnbd(x, t_x, T)
    assert m.r == pytest.approx(0.3, rel=0.3)
    assert m.alpha == pytest.approx(10.0, rel=0.4)
    assert m.r / m.alpha == pytest.approx(0.03, rel=0.2)  # mean purchase rate
    pa = m.p_alive(x, t_x, T)
    assert ((pa > 0) & (pa <= 1)).all() and (pa[x == 0] == 1).all()
    e = m.expected_purchases(26, x, t_x, T)
    assert (e >= 0).all() and e[x > 3].mean() > e[x == 0].mean()


def test_gamma_gamma_recovers_mean_spend():
    rng = np.random.default_rng(2)
    p, q, v = 3.0, 4.0, 150.0
    nu = rng.gamma(q, 1 / v, 3000)
    x = rng.integers(1, 6, 3000)
    m = np.array([rng.gamma(p, 1 / nu_i, k).mean() for nu_i, k in zip(nu, x, strict=True)])
    gg = clv.fit_gamma_gamma(x, m)
    assert gg.p * gg.v / (gg.q - 1) == pytest.approx(p * v / (q - 1), rel=0.15)
    spend = gg.expected_spend(np.array([0, 3]), np.array([np.nan, 200.0]))
    assert spend[0] == pytest.approx(gg.p * gg.v / (gg.q - 1))

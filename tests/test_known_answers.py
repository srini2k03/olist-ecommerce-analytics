"""Known-answer tests: exact values on the hand-made dataset in tests/mini_data.py.

These pin the definitions of the metric layer: what counts as a valid order, the money
columns, cohorts and repeat purchases, delivery days and lateness, and review averages.
"""

from __future__ import annotations

import pandas as pd
import pytest

from olist.sql_runner import list_queries, run

BY_NAME = {q.name: q for q in list_queries()}


def q(engine, name: str) -> pd.DataFrame:
    return run(engine, BY_NAME[name])


@pytest.fixture(scope="module")
def orders(mini):
    return mini.query("SELECT * FROM fct_orders").set_index("order_id")


def test_valid_orders_exclude_cancelled_and_itemless(orders):
    assert orders.is_valid.to_dict() == {"o1": 1, "o2": 1, "o3": 1, "o4": 1, "o5": 0, "o6": 0}
    assert orders.loc["o5", "is_cancelled"] == 1 and orders.loc["o6", "is_cancelled"] == 1


def test_money_columns_are_kept_apart(orders):
    o1 = orders.loc["o1"]
    assert (o1.item_revenue, o1.freight_value, o1.gmv) == (150.0, 15.0, 165.0)
    assert o1.payment_value == 170.0 and o1.payment_minus_gmv == pytest.approx(5.0)
    assert orders.loc["o2", "payment_value"] == 220.0  # voucher 20 + card 200
    assert orders.loc["o2", "used_voucher"] == 1 and orders.loc["o2", "main_payment_type"] == (
        "credit_card"
    )
    assert pd.isna(orders.loc["o6", "gmv"])  # no items: no GMV
    assert orders.loc["o4", "installments"] == 1  # 0 instalments fixed in staging


def test_main_seller_category_and_multi_seller_count(orders):
    assert orders.loc["o1", "main_seller_id"] == "s1"
    assert orders.loc["o1", "n_sellers"] == 2
    assert orders.loc["o1", "category"] == "health_beauty"
    assert orders.loc["o2", "category"] == "pc_gamer"  # missing from the translation table
    assert orders.loc["o4", "category"] == "unknown"  # product without a category


def test_customer_sequence_and_cohorts(orders):
    assert orders.loc["o1", "customer_order_seq"] == 1
    assert orders.loc["o2", "customer_order_seq"] == 2
    assert orders.loc["o2", "is_repeat_order"] == 1
    assert pd.Timestamp(orders.loc["o2", "cohort_month"]) == pd.Timestamp("2017-01-01")
    assert pd.Timestamp(orders.loc["o4", "cohort_month"]) == pd.Timestamp("2017-02-01")
    assert pd.isna(orders.loc["o6", "customer_order_seq"])  # invalid orders are not counted


def test_delivery_days_delay_and_lateness(orders):
    assert orders.loc["o1", ["delivery_days", "delay_days", "is_late"]].tolist() == [10, -5, 0]
    assert orders.loc["o2", ["delivery_days", "delay_days", "is_late"]].tolist() == [25, 5, 1]
    assert orders.loc["o3", ["delivery_days", "delay_days", "is_late"]].tolist() == [12, -9, 0]
    assert pd.isna(orders.loc["o4", "is_late"]) and orders.loc["o4", "is_delivered"] == 0
    assert orders.loc["o1", "estimated_days"] == 15
    assert orders.loc["o1", "handling_days"] == 2


def test_reviews_average_and_shared_review_ids(orders):
    assert orders.loc["o2", "review_score"] == 2.0 and orders.loc["o2", "n_reviews"] == 2
    assert orders.loc["o4", "review_score"] == 2.0  # review_id r4 is shared with o3
    assert orders.loc["o3", "review_score"] == 4.0


def test_zip_padding_and_distance(mini, orders):
    zips = mini.query("SELECT customer_id, zip_prefix FROM stg_customers")
    assert dict(zip(zips.customer_id, zips.zip_prefix, strict=True))["c1a"] == "01234"
    geo = mini.query("SELECT * FROM stg_geolocation WHERE zip_prefix = '01234'").iloc[0]
    assert (geo.lat, geo.lng, geo.n_points) == (pytest.approx(-23.6), pytest.approx(-46.7), 2)
    assert orders.loc["o1", "distance_km"] == pytest.approx(6.0, abs=0.5)
    assert orders.loc["o3", "distance_km"] == pytest.approx(355, abs=15)
    assert pd.isna(orders.loc["o4", "distance_km"])  # customer zip has no coordinates


def test_dim_customers(mini):
    d = mini.query("SELECT * FROM dim_customers").set_index("customer_unique_id")
    assert len(d) == 4
    assert d.loc["u1", "n_orders"] == 2 and d.loc["u1", "lifetime_gmv"] == 385.0
    assert d.loc["u1", "is_repeat_customer"] == 1
    assert d.loc["u4", "n_orders"] == 0 and pd.isna(d.loc["u4", "cohort_month"])


def test_monthly_revenue_query(mini):
    m = q(mini, "01_monthly_revenue_mom").set_index("month")
    jan = m.loc["2017-01-01"]
    assert (jan.orders, jan.item_revenue, jan.freight_value, jan.gmv) == (2, 230, 23, 253)
    assert m.loc["2017-02-01", "mom_growth"] == pytest.approx(44 / 253 - 1)


def test_repeat_rate_query(mini):
    rep = q(mini, "10_repeat_purchase_rate").set_index("cohort_quarter")
    assert rep.loc["All", "customers"] == 3
    assert rep.loc["All", "repeat_customers"] == 1
    assert rep.loc["All", "repeat_rate_90d"] == pytest.approx(1 / 3)


def test_order_funnel_query(mini):
    f = q(mini, "17_order_status_funnel").set_index("month")
    feb = f.loc["2017-02-01"]
    assert (feb.orders_placed, feb.orders_cancelled, feb.orders_delivered) == (2, 1, 0)
    assert feb.orders_in_progress == 1 and feb.cancellation_rate == 0.5
    assert f.loc["2017-04-01", "orders_cancelled"] == 1


def test_money_reconciliation_query(mini):
    r = q(mini, "18_money_reconciliation").set_index("month").loc["2017-01-01"]
    assert (r.gmv, r.payment_value, r.payment_minus_gmv) == (253, 258, 5)
    assert r.orders_payment_differs == 1


def test_review_by_delivery_status_query(mini):
    r = q(mini, "25_review_by_delivery_status").set_index("delivery_status")
    assert r.loc["on time", "orders"] == 2 and r.loc["on time", "avg_review"] == 4.5
    assert r.loc["late", "avg_review"] == 2.0
    assert r.loc["not delivered", "avg_review"] == 2.0


def test_cohort_retention_query(mini):
    c = q(mini, "02_cohort_retention")
    jan = c[c.cohort_month == "2017-01-01"].set_index("months_since_first")
    assert jan.loc[0, "cohort_size"] == 2  # u1 and u2
    assert jan.loc[2, "active_customers"] == 1 and jan.loc[2, "retention"] == 0.5


def test_seller_scorecard_counts_each_order_once_per_seller(mini):
    s = q(mini, "21_seller_scorecard").set_index("seller_id")
    assert s.loc["s1", "orders_placed"] == 4  # o1, o3, o4, o5
    assert s.loc["s1", "orders"] == 3 and s.loc["s1", "cancelled_orders"] == 1
    assert s.loc["s1", "item_revenue"] == 220  # 100 + 80 + 40 (o5 is cancelled)
    assert s.loc["s2", "late_orders"] == 1 and s.loc["s2", "delivered_orders"] == 2

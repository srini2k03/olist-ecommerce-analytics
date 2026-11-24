"""Metric layer: the YAML spec is complete and the pandas implementation matches the SQL."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from olist.config import METRICS_SPEC
from olist.metrics import (
    DOMAINS,
    KINDS,
    REGISTRY,
    UNITS,
    assumption_values,
    compute,
    compute_all,
    format_value,
    load_spec,
    metric_specs,
    metric_sql,
)

SPEC = load_spec(METRICS_SPEC)
SPECS = metric_specs(SPEC)
ASSUME = assumption_values(SPEC)


def test_spec_is_complete_and_consistent():
    names = [m["name"] for m in SPEC["metrics"]]
    assert len(names) == len(set(names))
    assert set(names) == set(REGISTRY), "every metric needs a pandas implementation"
    for s in SPECS.values():
        assert s.unit in UNITS and s.kind in KINDS and s.domain in DOMAINS, s.name
        assert len(s.definition) > 15 and s.sql, s.name
        if s.kind == "estimate":
            assert s.assumptions and set(s.assumptions) <= set(ASSUME), s.name
            assert "estimate" in s.label.lower()
    for a in SPEC["assumptions"].values():
        assert a["source"] == "assumption" and a["note"]


def test_money_metrics_are_distinct_definitions():
    money = {n: s for n, s in SPECS.items() if s.unit == "BRL"}
    exprs = [s.sql.get("expr", s.sql.get("query")) for s in money.values()]
    assert len(exprs) == len(set(exprs))
    assert {"item_revenue", "freight_value", "gmv", "payment_value"} <= set(money)


def _frames(engine):
    return (
        engine.query("SELECT * FROM fct_orders"),
        engine.query("SELECT * FROM fct_order_items"),
    )


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_pandas_matches_sql_on_synthetic_data(duck, name):
    orders, items = _frames(duck)
    sql_value = duck.query(metric_sql(SPECS[name], ASSUME))["value"].iloc[0]
    py_value = compute(name, orders, items, ASSUME)
    assert py_value == pytest.approx(float(sql_value), rel=1e-9, abs=1e-9), name


def test_known_values_on_mini_dataset(mini):
    orders, items = _frames(mini)
    m = compute_all(orders, items, ASSUME)
    assert m["gmv"] == 517.0  # 165 + 220 + 88 + 44
    assert m["item_revenue"] == 470.0 and m["freight_value"] == 47.0
    assert m["payment_value"] == 522.0  # 170 + 220 + 88 + 44
    assert m["orders"] == 4 and m["orders_placed"] == 6
    assert m["aov"] == pytest.approx(129.25)
    assert m["cancellation_rate"] == pytest.approx(2 / 6)
    assert m["delivery_completion_rate"] == pytest.approx(3 / 6)
    assert (m["customers"], m["new_customers"], m["returning_customers"]) == (3, 3, 1)
    assert m["repeat_rate"] == pytest.approx(1 / 3)
    assert m["late_rate"] == pytest.approx(1 / 3)
    assert m["avg_delivery_days"] == pytest.approx((10 + 25 + 12) / 3)
    assert m["avg_review"] == pytest.approx((5 + 2 + 4 + 2) / 4)
    assert m["review_gap_late"] == pytest.approx(4.5 - 2.0)
    assert m["marketplace_revenue_estimate"] == pytest.approx(ASSUME["take_rate"] * 470)
    assert m["active_sellers"] == 2 and m["active_categories"] == 3


def test_item_filter_sums_matching_items_only(mini):
    orders, items = _frames(mini)
    sel = items[items.seller_id == "s2"]
    o = orders[orders.order_id.isin(sel.order_id)]
    assert compute("item_revenue", o, sel) == 250.0  # o1 item 2 (50) + o2 (200)
    assert compute("orders", o, sel) == 2
    assert compute("gmv", o, sel) == 275.0


def test_metrics_on_empty_selection_are_nan_not_errors(mini):
    orders, items = _frames(mini)
    m = compute_all(orders.iloc[0:0], items.iloc[0:0], ASSUME)
    assert m["orders"] == 0 and m["gmv"] == 0
    assert math.isnan(m["aov"]) and math.isnan(m["late_rate"])


@pytest.mark.parametrize(
    ("value", "unit", "text"),
    [
        (15_688_396.6, "BRL", "R$ 15.69M"),
        (160.24, "BRL", "R$ 160.24"),
        (0.0679, "share", "6.8%"),
        (12.04, "days", "12.0 days"),
        (4.0912, "stars", "4.09★"),
        (97905.0, "count", "97,905"),
        (float("nan"), "share", "n/a"),
    ],
)
def test_format_value(value, unit, text):
    assert format_value(value, unit) == text


def test_spec_round_trips_through_json(tmp_path):
    import json

    p = tmp_path / "metrics.json"
    p.write_text(json.dumps(SPEC))
    assert metric_specs(load_spec(p)) == SPECS
    assert isinstance(pd.DataFrame(), pd.DataFrame)

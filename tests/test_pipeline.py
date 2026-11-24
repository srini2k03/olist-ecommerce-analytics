"""The layered pipeline: model contracts, data-quality checks and failure behaviour."""

from __future__ import annotations

import mini_data
import pandas as pd
import pytest

from olist.db import SchemaError
from olist.pipeline.checks import DataQualityError
from olist.pipeline.models import build_order, list_models
from olist.pipeline.run import render_markdown, run_duckdb

MODELS = list_models()


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.name)
def test_every_model_documents_layer_grain_purpose_and_dependencies(model):
    h = model.header
    for key in ("Layer", "Grain", "Purpose", "Depends on"):
        assert key in h, f"{model.name} lacks '{key}:'"
    assert h["Layer"] == model.layer
    assert model.depends_on


def test_build_order_respects_dependencies():
    order = [m.name for m in build_order(MODELS)]
    for m in MODELS:
        for dep in m.depends_on:
            if dep in order:
                assert order.index(dep) < order.index(m.name), (dep, m.name)
    assert order.index("fct_orders") < order.index("dim_customers")


def test_marts_only_read_staging_or_marts():
    """Analytics should never reach past the staging layer into raw tables."""
    staging = {m.name for m in MODELS if m.layer == "staging"}
    for m in MODELS:
        if m.layer == "marts":
            assert set(m.depends_on) <= staging | {x.name for x in MODELS}, m.name


def test_mini_dataset_passes_with_expected_warnings(mini_report):
    _, report = mini_report
    assert report["summary"]["fail"] == 0
    by_name = {c["name"]: c for c in report["checks"]}
    assert by_name["orders has order_items"]["failing"] == 1  # o6
    assert by_name["orders has order_items"]["status"] == "warn"
    assert by_name["order_items.order_id -> orders.order_id"]["status"] == "pass"
    assert by_name["fct_orders: payment close to gmv"]["failing"] == 1  # o1 interest
    assert report["row_counts"]["marts"]["fct_orders"] == 6
    assert report["row_counts"]["staging"]["stg_geolocation"] == 4  # outlier dropped
    assert "Data-quality report" in render_markdown(report)


def test_orphan_item_fails_referential_integrity(tmp_path):
    items = mini_data.tables()["order_items"]
    orphan = pd.DataFrame(
        [("o_missing", 1, "p1", "s1", pd.Timestamp("2017-01-12"), 10.0, 1.0)],
        columns=items.columns,
    )
    raw = mini_data.write_csvs(
        tmp_path / "raw", {"order_items": pd.concat([items, orphan], ignore_index=True)}
    )
    with pytest.raises(DataQualityError, match="order_items.order_id -> orders.order_id"):
        run_duckdb(tmp_path / "wh.duckdb", raw, report_dir=None)


def test_duplicate_order_id_fails(tmp_path):
    orders = mini_data.tables()["orders"]
    raw = mini_data.write_csvs(
        tmp_path / "raw", {"orders": pd.concat([orders, orders.head(1)], ignore_index=True)}
    )
    with pytest.raises(DataQualityError, match=r"orders\(order_id\) unique"):
        run_duckdb(tmp_path / "wh.duckdb", raw, report_dir=None)


def test_out_of_range_review_score_fails(tmp_path):
    reviews = mini_data.tables()["order_reviews"].copy()
    reviews.loc[0, "review_score"] = 7
    raw = mini_data.write_csvs(tmp_path / "raw", {"order_reviews": reviews})
    with pytest.raises(DataQualityError, match="score 1-5"):
        run_duckdb(tmp_path / "wh.duckdb", raw, report_dir=None)


def test_missing_column_is_a_schema_error(tmp_path):
    sellers = mini_data.tables()["sellers"].drop(columns=["seller_state"])
    raw = mini_data.write_csvs(tmp_path / "raw")
    sellers.to_csv(raw / "olist_sellers_dataset.csv", index=False)
    with pytest.raises(SchemaError, match="seller_state"):
        run_duckdb(tmp_path / "wh.duckdb", raw, report_dir=None)


def test_unparseable_timestamp_is_counted(tmp_path):
    orders = mini_data.tables()["orders"].astype({"order_approved_at": object})
    orders.loc[0, "order_approved_at"] = "not a date"
    raw = mini_data.write_csvs(tmp_path / "raw", {"orders": orders})
    report = run_duckdb(tmp_path / "wh.duckdb", raw, report_dir=None)
    assert report["timestamp_parse_failures"] == {"orders.order_approved_at": 1}


def test_synthetic_warehouse_has_no_failing_checks(synthetic_duckdb):
    from olist.db import DuckDBEngine
    from olist.pipeline.checks import run_checks

    eng = DuckDBEngine(synthetic_duckdb)
    try:
        for layer in ("raw", "staging", "marts"):
            assert not [r.name for r in run_checks(eng, layer) if r.status == "fail"]
    finally:
        eng.close()

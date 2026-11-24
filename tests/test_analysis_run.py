"""End-to-end: every analysis runs on the synthetic warehouse and writes its outputs."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from olist.analysis import run_all
from olist.analytics import delivery_model


@pytest.fixture(scope="module")
def outputs(synthetic_duckdb, synthetic_raw, tmp_path_factory):
    out = tmp_path_factory.mktemp("analytics")
    run_all(out, synthetic_duckdb, synthetic_raw)
    return out


def test_every_output_is_written_with_provenance(outputs):
    expected = {
        "findings.json",
        "metrics_snapshot.json",
        "late_vs_on_time.json",
        "cohorts.csv",
        "retention_curve.csv",
        "cohort_summary.csv",
        "rfm_segments.csv",
        "clv.json",
        "clv_by_segment.csv",
        "seller_ranking.csv",
        "seller_ranking.json",
        "delivery_model.json",
        "ab_test.json",
        "scenarios.json",
    }
    assert expected <= {p.name for p in outputs.iterdir()}
    for p in outputs.glob("*.json"):
        meta = json.loads(p.read_text())["run_metadata"]
        assert meta["data_fingerprint"] and meta["versions"]["python"]


def test_simulated_outputs_are_labelled(outputs):
    ab = json.loads((outputs / "ab_test.json").read_text())
    assert ab["result"]["simulated"] is True and "SIMULATED" in ab["label"]
    sc = json.loads((outputs / "scenarios.json").read_text())["scenarios"]
    assert all(s["hypothetical"] for s in sc.values())


def test_delivery_model_uses_only_past_information(synthetic_duckdb):
    from olist.db import DuckDBEngine

    eng = DuckDBEngine(synthetic_duckdb)
    fo = eng.query("SELECT * FROM fct_orders")
    eng.close()
    f = delivery_model.build_features(fo)
    # A seller's history at an order never includes deliveries at or after its purchase time.
    delivered = fo[fo.is_delivered == 1][["main_seller_id", "delivered_at"]]
    sample = f.dropna(subset=["main_seller_id"]).sample(200, random_state=0)
    for _, row in sample.iterrows():
        d = delivered[delivered.main_seller_id == row.main_seller_id]
        expected = int((pd.to_datetime(d.delivered_at) < row.purchased_at).sum())
        assert row.seller_past_orders == expected


def test_delivery_model_splits_are_time_ordered(outputs):
    dm = json.loads((outputs / "delivery_model.json").read_text())
    s = dm["splits"]
    assert s["train"]["to_exclusive"] <= s["valid"]["from"] < s["valid"]["to_exclusive"]
    assert s["valid"]["to_exclusive"] <= s["test"]["from"]
    assert set(dm["regression_delivery_days"]) == {"olist_promise", "lane_history_mean", "lightgbm"}

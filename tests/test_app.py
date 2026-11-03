"""Smoke-test the Streamlit app against the committed artefacts in app/data."""

from pathlib import Path

import pandas as pd
import pytest

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
DATA = APP.parent / "data"

pytestmark = pytest.mark.skipif(
    not (DATA / "churn_scores.parquet").exists(), reason="app/data not built"
)


def test_artefacts_are_small_and_complete():
    expected = {
        "kpis.json",
        "model_card.json",
        "monthly.parquet",
        "cohort.parquet",
        "delivery_delay.parquet",
        "state_sla.parquet",
        "segments.parquet",
        "segment_sample.parquet",
        "churn_scores.parquet",
    }
    assert expected <= {p.name for p in DATA.iterdir()}
    total = sum(p.stat().st_size for p in DATA.iterdir())
    assert total < 10 * 1024 * 1024, f"app/data is {total / 1e6:.1f} MB"


def test_app_renders_without_exceptions():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception, at.exception
    assert at.title[0].value == "Olist marketplace analytics"
    assert len(at.metric) >= 6
    assert len(at.tabs) == 5


def test_churn_lookup_finds_a_customer_and_shows_reasons():
    from streamlit.testing.v1 import AppTest

    cid = pd.read_parquet(DATA / "churn_scores.parquet").customer_unique_id.iloc[123]
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    at.text_input[0].set_value(cid).run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "P(repeat in 180 days)" in labels
    assert any("churn risk" in md.value for md in at.markdown)


def test_unknown_customer_shows_error():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=120).run()
    at.text_input[0].set_value("not-a-customer").run()
    assert at.error and "not found" in at.error[0].value

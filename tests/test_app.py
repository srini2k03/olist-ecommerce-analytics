"""The 10-page dashboard against the committed bundle in app/data."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
APP = APP_DIR / "streamlit_app.py"
DATA = APP_DIR / "data"

pytestmark = pytest.mark.skipif(
    not (DATA / "orders.parquet").exists(), reason="app/data bundle not built"
)
sys.path.insert(0, str(APP_DIR))

from dashboard.app import PAGES, TITLE  # noqa: E402

HEAVY = ("scipy", "sklearn", "lightgbm", "duckdb", "statsmodels", "yaml", "shap", "psycopg")


def page_script(title: str, setup: str = "", teardown: str = "pass") -> str:
    return (
        f"import sys\nsys.path.insert(0, {str(APP_DIR)!r})\n"
        f"sys.path.insert(0, {str(ROOT / 'src')!r})\n{setup}\n"
        f"from dashboard.app import main\ntry:\n    main(only_page={title!r})\n"
        f"finally:\n    {teardown}\n"
    )


def run_page(title: str, setup: str = "", teardown: str = "pass"):
    from streamlit.testing.v1 import AppTest

    script = page_script(title, setup, teardown)
    return AppTest.from_string(script, default_timeout=300).run()


def test_bundle_is_complete_and_small():
    expected = {
        "orders.parquet", "items.parquet", "sellers.parquet", "metrics.json",
        "data_quality.json", "findings.json", "clv.json", "delivery_model.json",
        "metrics_snapshot.json", "segments.parquet", "segment_sample.parquet",
        "churn_scores.parquet", "model_card.json",
    }  # fmt: skip
    names = {p.name for p in DATA.iterdir()}
    assert expected <= names, expected - names
    total = sum(p.stat().st_size for p in DATA.iterdir())
    assert total < 9 * 1024 * 1024, f"app/data is {total / 1e6:.1f} MB"


def test_bundle_metrics_match_the_warehouse_snapshot():
    """The dashboard's own computation on the bundle (default filters) reproduces the metric
    snapshot computed from the marts by scripts/run_analysis.py."""
    from dashboard import data

    from olist.metrics import assumption_values, compute_all

    o, it = data.apply(data.Filters())
    snap = json.loads((DATA / "metrics_snapshot.json").read_text())["metrics"]
    ours = compute_all(o, it, assumption_values(json.loads((DATA / "metrics.json").read_text())))
    for name, s in snap.items():
        if s["value"] is None:
            continue
        assert ours[name] == pytest.approx(s["value"], rel=1e-4, abs=1e-6), name


@pytest.mark.parametrize("title", [p[0] for p in PAGES])
def test_every_page_renders(title):
    at = run_page(title)
    assert not at.exception, [e.value for e in at.exception]
    assert at.title, title
    expected = TITLE if title == "Executive overview" else title
    assert at.title[0].value == expected


def test_entry_point_uses_navigation_and_opens_the_overview():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=300).run()
    assert not at.exception
    assert at.title[0].value == TITLE
    assert len(at.metric) >= 12


def _metric(at, label: str) -> str:
    return next(m.value for m in at.metric if m.label == label)


def test_filters_change_the_numbers():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(page_script("Executive overview"), default_timeout=300).run()
    gmv_all, orders_all = _metric(at, "GMV"), _metric(at, "Orders")
    at.multiselect(key="f_cstate").set_value(["SP"]).run()
    assert not at.exception
    assert _metric(at, "GMV") != gmv_all and _metric(at, "Orders") != orders_all
    cat = sorted(pd.read_parquet(DATA / "items.parquet").category.dropna().astype(str))[0]
    at.multiselect(key="f_cat").set_value([cat]).run()
    assert not at.exception


@pytest.mark.parametrize("title", [p[0] for p in PAGES])
def test_pages_show_an_empty_state(title):
    setup = (
        "import streamlit as st\n"
        "from dashboard import data\n"
        "_apply = data.apply\n"
        "data.apply = lambda f: tuple(x.iloc[0:0] for x in _apply(f))\n"
    )
    at = run_page(title, setup, teardown="data.apply = _apply")
    assert not at.exception, [e.value for e in at.exception]
    assert any("No orders match" in i.value for i in at.info), title


def test_churn_lookup_finds_a_customer_and_explains():
    cid = pd.read_parquet(DATA / "churn_scores.parquet").customer_unique_id.iloc[123]
    at = run_page("Customers")
    at.text_input(key="churn_id").set_value(cid).run()
    assert not at.exception
    assert "P(repeat in 180 days)" in [m.label for m in at.metric]
    assert any("churn risk" in md.value for md in at.markdown)


def test_unknown_customer_shows_error():
    at = run_page("Customers")
    at.text_input(key="churn_id").set_value("not-a-customer").run()
    assert at.error and "not found" in at.error[0].value


def test_ab_page_reports_a_decision_and_labels_the_simulation():
    at = run_page("A/B testing")
    assert not at.exception
    heads = [h.value for h in at.subheader]
    assert any(h.startswith("Decision engine:") for h in heads)
    assert any("Simulated" in m.value for m in at.markdown)


def test_browser_bundle_has_no_heavy_dependencies():
    """The stlite build ships only build_site.APP_MODULES; the dashboard must import nothing
    else from olist, and nothing heavy at all."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_site

    code = f"""
import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in {HEAVY!r}:
            raise ImportError('blocked in the browser bundle: ' + name)
sys.meta_path.insert(0, Block())
sys.path[:0] = [{str(APP_DIR)!r}, {str(ROOT / "src")!r}]
import dashboard.app
print(sorted(m for m in sys.modules if m == 'olist' or m.startswith('olist.')))
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    loaded = eval(out.stdout.strip().splitlines()[-1])  # noqa: S307 (our own subprocess)
    shipped = {
        m.removesuffix(".py").replace("/", ".").removesuffix(".__init__")
        for m in build_site.APP_MODULES
    }
    assert set(loaded) <= shipped, set(loaded) - shipped


def test_pages_workflow_opens_every_page():
    text = (ROOT / ".github" / "workflows" / "pages.yml").read_text()
    listed = text.split('--pages "', 1)[1].split('"', 1)[0].split(",")
    assert sorted(listed) == sorted(p[0] for p in PAGES)

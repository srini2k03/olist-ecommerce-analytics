"""The generated README section: pending until real results exist, never synthetic."""

from __future__ import annotations

import json
import shutil

from recsys import config, readme


def test_committed_readme_section_is_current():
    assert readme.is_current(), "run `python -m recsys readme`"


def test_committed_results_folder_has_no_synthetic_results():
    for path in config.RESULTS_DIR.glob("*.json"):
        assert json.loads(path.read_text()).get("data") == "real", path


def test_pending_section_cites_the_olist_numbers(tmp_path):
    text = readme.render(tmp_path)
    assert "**Status: pending.**" in text and "run recsys-train" in text
    assert "3.04%" in text and "90%" in text and "```mermaid" in text
    assert text.startswith(readme.START) and text.endswith(readme.END)


def test_synthetic_results_are_ignored(synthetic_pipeline):
    text = readme.render(synthetic_pipeline / "results")
    assert "**Status: pending.**" in text


def test_real_results_fill_the_section(synthetic_pipeline, tmp_path):
    for name in ("report.json", "serving_latency.json"):
        data = json.loads((synthetic_pipeline / "results" / name).read_text())
        data["data"] = "real"  # pretend, to exercise the rendering
        (tmp_path / name).write_text(json.dumps(data))
    text = readme.render(tmp_path)
    assert "pending" not in text.lower()
    for heading in (
        "### All stages",
        "### Ranker feature ablation",
        "### Cold start",
        "### Simulated A/B test",
    ):
        assert heading in text
    assert "**p95" in text and "Decision:" in text


def test_update_rewrites_only_the_marked_section(tmp_path):
    shutil.copy(readme.README, tmp_path / "README.md")
    path = tmp_path / "README.md"
    text = path.read_text()
    broken = text.replace("## Fashion recommender (H&M data)", "## stale")
    path.write_text(broken)
    assert not readme.is_current(path)
    assert readme.update(path) is True
    assert path.read_text() == text
    assert readme.update(path) is False

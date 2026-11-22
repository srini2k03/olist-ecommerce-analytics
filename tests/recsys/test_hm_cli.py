"""End-to-end CLI on synthetic data (no network)."""

from __future__ import annotations

import json

import pytest

from recsys.cli import main


def test_synthetic_pipeline_end_to_end(tmp_path, capsys):
    assert main(["--synthetic", "--data-dir", str(tmp_path), "--n-boot", "50"]) == 0
    for sub in ("parquet", "subset", "split"):
        assert (tmp_path / sub).is_dir()
    report = json.loads((tmp_path / "split" / "split_report.json").read_text())
    assert report["leakage_checks"] == "passed"
    assert report["cold_start"]["test"]["cold_users"] > 0
    text = (tmp_path / "results" / "baselines_validation.json").read_text()
    assert "NaN" not in text
    result = json.loads(text)
    assert result["data"] == "synthetic"
    assert result["split"] == "validation"
    assert set(result["baselines"]) == {
        "popularity_1w",
        "age_popularity_4w",
        "item_cooccurrence",
        "repurchase",
    }
    maps = {
        r["model"]: r["value"] for r in result["metrics"] if r["metric"] == "map" and r["k"] == 12
    }
    assert all(0 < v < 1 for v in maps.values())
    assert {r["k"] for r in result["metrics"]} == {12, 50, 100}
    coverage = [r for r in result["metrics"] if r["metric"] == "coverage"]
    assert coverage and all(r["ci_low"] is None for r in coverage)  # strict JSON, no NaN
    out = capsys.readouterr().out
    assert "map@12" in out and "repurchase" in out

    # Same seed, same numbers.
    other = tmp_path / "again"
    assert main(["--synthetic", "--data-dir", str(other), "--n-boot", "50"]) == 0
    again = json.loads((other / "results" / "baselines_validation.json").read_text())
    assert again["metrics"] == result["metrics"]


def test_fetch_without_credentials_fails_cleanly(tmp_path, monkeypatch, caplog):
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path))
    code = main(["--data-dir", str(tmp_path), "fetch", "--raw-dir", str(tmp_path / "raw")])
    assert code == 1
    assert "KAGGLE_USERNAME" in caplog.text
    assert not (tmp_path / "raw").exists()


def test_synthetic_flag_rejects_a_command():
    with pytest.raises(SystemExit):
        main(["--synthetic", "split"])

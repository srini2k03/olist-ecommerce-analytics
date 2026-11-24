"""Generated docs are complete and up to date."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_docs  # noqa: E402


def test_every_mart_column_is_documented(synthetic_duckdb):
    doc = yaml.safe_load((ROOT / "sql" / "marts" / "columns.yml").read_text())
    missing = [
        f"{table}.{col}"
        for table, cols in build_docs.mart_columns(synthetic_duckdb).items()
        if table.startswith(("fct_", "dim_"))
        for col, _ in cols
        if not build_docs.column_description(doc, table, col)
    ]
    assert not missing, missing


def test_generated_docs_are_committed_and_current(synthetic_duckdb):
    for name, text in build_docs.generate(synthetic_duckdb).items():
        committed = (ROOT / "docs" / name).read_text(encoding="utf-8")
        assert committed == text, f"docs/{name} is stale: run python scripts/build_docs.py"


def test_docs_link_targets_exist():
    import re

    for md in (ROOT / "docs").glob("*.md"):
        for target in re.findall(r"\]\((\.\./[^)#]+|[a-z_]+\.md)\)", md.read_text()):
            assert (md.parent / target).resolve().exists(), f"{md.name} -> {target}"

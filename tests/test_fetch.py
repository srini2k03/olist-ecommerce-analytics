import importlib.util
from pathlib import Path

from olist.schema import TABLES

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_data.py"


def _load():
    spec = importlib.util.spec_from_file_location("fetch_data", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_manifest_covers_all_nine_tables_with_checksums():
    mod = _load()
    assert set(mod.FILES) == {t.csv for t in TABLES}
    for sha, size in mod.FILES.values():
        assert len(sha) == 64 and size > 0


def test_mirror_is_pinned_to_a_commit():
    mod = _load()
    assert len(mod.MIRROR_COMMIT) == 40
    assert mod.MIRROR_COMMIT in mod.MIRROR_BASE


def test_is_valid_rejects_wrong_bytes(tmp_path):
    mod = _load()
    p = tmp_path / "olist_sellers_dataset.csv"
    p.write_text("seller_id\n")
    assert not mod.is_valid(p)

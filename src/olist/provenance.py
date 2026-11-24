"""Provenance for generated results: library versions, seed and a fingerprint of the input
data. Deliberately no timestamp or git SHA inside the files, so that a rebuild from the same
code and data is byte-identical; the commit that adds a result file records when and from
which code it was produced."""

from __future__ import annotations

import hashlib
import platform
from importlib import metadata
from pathlib import Path

from olist.config import RANDOM_SEED, RAW_DIR

PACKAGES = ("duckdb", "pandas", "numpy", "scipy", "scikit-learn", "lightgbm", "statsmodels")


def data_fingerprint(raw_dir: Path = RAW_DIR) -> str:
    """SHA-256 over the names, sizes and first/last MiB of the raw CSVs (cheap, and enough
    to tell the pinned dataset from the synthetic test fixture or a different download)."""
    h = hashlib.sha256()
    for p in sorted(Path(raw_dir).glob("*.csv")):
        size = p.stat().st_size
        h.update(f"{p.name}:{size}".encode())
        with p.open("rb") as f:
            h.update(f.read(1 << 20))
            if size > 1 << 20:
                f.seek(-(1 << 20), 2)
                h.update(f.read())
    return h.hexdigest()[:16]


def versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return out


def run_metadata(raw_dir: Path = RAW_DIR, seed: int | None = RANDOM_SEED) -> dict:
    meta = {"data_fingerprint": data_fingerprint(raw_dir), "versions": versions()}
    if seed is not None:
        meta["seed"] = seed
    return meta

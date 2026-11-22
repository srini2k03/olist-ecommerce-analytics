"""Download the H&M competition tables with the Kaggle CLI and convert them to parquet.

Only ``transactions_train.csv``, ``articles.csv`` and ``customers.csv`` are
downloaded (the images are ~30 GB and not needed). The raw files stay under the
gitignored ``data/hm/raw/`` because the competition rules do not allow
redistribution. A ``manifest.json`` records each file's size and SHA-256 so a
later run can tell whether the inputs changed.

Requires a Kaggle account that has accepted the competition rules, with
credentials in ``KAGGLE_USERNAME``/``KAGGLE_KEY`` or ``~/.kaggle/kaggle.json``.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import zipfile
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from recsys.config import COMPETITION, RAW_FILES
from recsys.data.schema import (
    ARTICLE_DTYPES,
    CUSTOMER_DTYPES,
    TRANSACTION_DTYPES,
    coerce,
    hash_customer_series,
    write_json,
)

log = logging.getLogger(__name__)

MANIFEST = "manifest.json"
CHUNK_ROWS = 2_000_000

Runner = Callable[..., subprocess.CompletedProcess]


class MissingCredentialsError(RuntimeError):
    """Raised when no Kaggle credentials are configured."""


def credentials_available(env: Mapping[str, str] | None = None) -> bool:
    env = os.environ if env is None else env
    if env.get("KAGGLE_USERNAME") and env.get("KAGGLE_KEY"):
        return True
    config_dir = Path(env.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    return (config_dir / "kaggle.json").is_file()


def require_credentials(env: Mapping[str, str] | None = None) -> None:
    if not credentials_available(env):
        raise MissingCredentialsError(
            "No Kaggle credentials found. Set KAGGLE_USERNAME and KAGGLE_KEY (or put "
            "kaggle.json in ~/.kaggle/), and accept the rules of the competition at "
            f"https://www.kaggle.com/competitions/{COMPETITION}/rules before downloading."
        )


def sha256_file(path: Path, block: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(raw_dir: Path, files: tuple[str, ...] = RAW_FILES) -> dict:
    manifest = {
        "competition": COMPETITION,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": {
            name: {"bytes": (raw_dir / name).stat().st_size, "sha256": sha256_file(raw_dir / name)}
            for name in files
        },
    }
    write_json(raw_dir / MANIFEST, manifest)
    return manifest


def _unzip_if_needed(raw_dir: Path, name: str) -> None:
    archive = raw_dir / f"{name}.zip"
    if archive.exists():
        with zipfile.ZipFile(archive) as zf:
            zf.extract(name, raw_dir)
        archive.unlink()


def download(
    raw_dir: Path,
    files: tuple[str, ...] = RAW_FILES,
    force: bool = False,
    runner: Runner = subprocess.run,
    env: Mapping[str, str] | None = None,
) -> dict:
    """Download ``files`` into ``raw_dir`` and write the manifest. Returns the manifest."""
    require_credentials(env)
    kaggle = shutil.which("kaggle")
    if kaggle is None:
        raise RuntimeError("kaggle CLI not found; pip install -r requirements-recsys.txt")
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in files:
        if (raw_dir / name).exists() and not force:
            log.info("%s already present, skipping download", name)
            continue
        cmd = [kaggle, "competitions", "download", COMPETITION, "-f", name, "-p", str(raw_dir)]
        if force:
            cmd.append("--force")
        log.info("downloading %s", name)
        runner(cmd, check=True)  # noqa: S603 (fixed argument list, no shell)
        _unzip_if_needed(raw_dir, name)
        if not (raw_dir / name).exists():
            raise RuntimeError(f"kaggle finished but {raw_dir / name} does not exist")
    manifest = write_manifest(raw_dir, files)
    for name, meta in manifest["files"].items():
        log.info("%s: %s bytes, sha256 %s", name, f"{meta['bytes']:,}", meta["sha256"][:12])
    return manifest


def _csv_dtypes(dtypes: dict[str, str]) -> dict[str, str]:
    """Dtypes to read a raw CSV with (customer ids as strings, dates parsed separately)."""
    out = {c: d for c, d in dtypes.items() if c not in {"customer_id", "t_dat"}}
    if "customer_id" in dtypes:
        out["customer_id"] = "str"
    return out


def convert_customers(raw_dir: Path, out_dir: Path) -> int:
    raw = pd.read_csv(raw_dir / "customers.csv", dtype=_csv_dtypes(CUSTOMER_DTYPES))
    hashed = hash_customer_series(raw["customer_id"])
    if pd.Series(hashed).duplicated().any():
        raise ValueError("customer_id hash collision; widen the hash before continuing")
    mapping = pd.DataFrame({"customer_id": hashed, "customer_id_hex": raw["customer_id"]})
    mapping.to_parquet(out_dir / "customer_id_map.parquet", index=False)
    raw["customer_id"] = hashed
    coerce(raw, "customers").to_parquet(out_dir / "customers.parquet", index=False)
    return len(raw)


def convert_articles(raw_dir: Path, out_dir: Path) -> int:
    raw = pd.read_csv(raw_dir / "articles.csv", dtype=_csv_dtypes(ARTICLE_DTYPES))
    coerce(raw, "articles").to_parquet(out_dir / "articles.parquet", index=False)
    return len(raw)


def convert_transactions(raw_dir: Path, out_dir: Path, chunk_rows: int = CHUNK_ROWS) -> int:
    """Stream the 3.5 GB CSV in chunks so peak memory stays small."""
    schema = pa.schema(
        [
            ("t_dat", pa.timestamp("ns")),
            ("customer_id", pa.int64()),
            ("article_id", pa.int32()),
            ("price", pa.float32()),
            ("sales_channel_id", pa.int8()),
        ]
    )
    rows = 0
    reader = pd.read_csv(
        raw_dir / "transactions_train.csv",
        dtype=_csv_dtypes(TRANSACTION_DTYPES),
        parse_dates=["t_dat"],
        chunksize=chunk_rows,
    )
    with pq.ParquetWriter(out_dir / "transactions.parquet", schema) as writer:
        for chunk in reader:
            chunk["customer_id"] = hash_customer_series(chunk["customer_id"])
            table = pa.Table.from_pandas(
                coerce(chunk, "transactions"), schema=schema, preserve_index=False
            )
            writer.write_table(table)
            rows += len(chunk)
    return rows


def convert(raw_dir: Path, out_dir: Path, chunk_rows: int = CHUNK_ROWS) -> dict[str, int]:
    """Convert the three raw CSVs into typed parquet files in ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {
        "customers": convert_customers(raw_dir, out_dir),
        "articles": convert_articles(raw_dir, out_dir),
        "transactions": convert_transactions(raw_dir, out_dir, chunk_rows),
    }
    for name, n in counts.items():
        log.info("%s: %s rows -> %s", name, f"{n:,}", out_dir / f"{name}.parquet")
    return counts

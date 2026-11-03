"""Download the nine Olist CSVs into data/raw and verify their checksums.

Source order:
1. Kaggle API (``olistbr/brazilian-ecommerce``) when ``KAGGLE_USERNAME`` and
   ``KAGGLE_KEY`` are set.
2. Olist's own public GitHub repository ``olist/work-at-olist-data``, pinned to
   a commit so the bytes never change.

Every file is checked against the SHA-256 values below. A Kaggle file that
does not match is replaced from the pinned mirror, so every run analyses the
same bytes.

The data is licensed CC BY-NC-SA 4.0 by Olist. It is downloaded at build time
and never committed to this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import sys
import urllib.request
import zipfile
from base64 import b64encode
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "data" / "raw"

MIRROR_COMMIT = "d9e49802f3e92d09ee94ab9ccc5e457f207a8959"
MIRROR_BASE = f"https://raw.githubusercontent.com/olist/work-at-olist-data/{MIRROR_COMMIT}/datasets"
KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/olistbr/brazilian-ecommerce"

# file name -> (sha256, size in bytes)
FILES: dict[str, tuple[str, int]] = {
    "olist_customers_dataset.csv": (
        "c26c17f59f3027a6a0dbeb8f1fa373c38f3323ecc84f225a8f4820e5bb288df8",
        8562270,
    ),
    "olist_geolocation_dataset.csv": (
        "b514f6fc991b9566aeba02aa5d67e2c3630f034b60a0e05aa0d082a3b66d88d6",
        61273883,
    ),
    "olist_order_items_dataset.csv": (
        "4f6abdbbc94036d0df4a76fa0520c072e31a40119d70f7f370fba1e2285d2bcb",
        15007623,
    ),
    "olist_order_payments_dataset.csv": (
        "61674868ed7b872aac3dcd95bf774448a88f6426b183f21be55f75af021ffdec",
        5647783,
    ),
    "olist_order_reviews_dataset.csv": (
        "012b61c7593e34f51fa614efdf802b9c7056ce6aae5307ddb93236e7cfc797d7",
        14451670,
    ),
    "olist_orders_dataset.csv": (
        "8df58ef3d2d7e9944010f7beecd9b75367f5588ec6e3c91cec19ae3345ef9ecf",
        17654914,
    ),
    "olist_products_dataset.csv": (
        "3e6569628a17fbc75fd206ee357b59e20364b9afa90f5b6cd5b4d624c58aa9cc",
        2379446,
    ),
    "olist_sellers_dataset.csv": (
        "31eb9bd50d684526a78f2389413ce61841e97692de7b828fe7d956b534586312",
        163589,
    ),
    "product_category_name_translation.csv": (
        "a81f0d1f27b27e7293f761bc79e3ce8f348ee39c4b3ed3e49bde38f478586278",
        2613,
    ),
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def is_valid(path: Path) -> bool:
    if not path.exists():
        return False
    expected, size = FILES[path.name]
    return path.stat().st_size == size and sha256(path.read_bytes()) == expected


def _get(url: str, headers: dict[str, str] | None = None, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers=headers or {})  # noqa: S310 (fixed https URLs)
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (fixed https URLs)
        return resp.read()


def fetch_kaggle(out: Path) -> list[str]:
    """Return the files written from Kaggle (only those that match the checksums)."""
    user, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if not (user and key):
        return []
    token = b64encode(f"{user}:{key}".encode()).decode()
    try:
        blob = _get(KAGGLE_URL, {"Authorization": f"Basic {token}"}, timeout=300)
    except Exception as exc:  # network or auth failure: fall back to the mirror
        print(f"[kaggle] download failed ({exc}); using the pinned mirror", file=sys.stderr)
        return []
    written = []
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for name in FILES:
            if name not in zf.namelist():
                continue
            data = zf.read(name)
            if sha256(data) != FILES[name][0]:
                print(f"[kaggle] {name} differs from the pinned version; using mirror")
                continue
            (out / name).write_bytes(data)
            written.append(name)
    print(f"[kaggle] {len(written)} files verified")
    return written


def fetch_mirror(out: Path, names: list[str]) -> None:
    for name in names:
        data = _get(f"{MIRROR_BASE}/{name}")
        got = sha256(data)
        if got != FILES[name][0]:
            raise SystemExit(f"checksum mismatch for {name}: {got}")
        (out / name).write_bytes(data)
        print(f"[mirror] {name} ({len(data):,} bytes) verified")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force", action="store_true", help="re-download valid files")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    todo = [n for n in FILES if args.force or not is_valid(args.out / n)]
    if not todo:
        print(f"all {len(FILES)} files present and verified in {args.out}")
        return 0
    done = set(fetch_kaggle(args.out))
    fetch_mirror(args.out, [n for n in todo if n not in done])
    bad = [n for n in FILES if not is_valid(args.out / n)]
    if bad:
        raise SystemExit(f"verification failed: {bad}")
    print(f"all {len(FILES)} files verified in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

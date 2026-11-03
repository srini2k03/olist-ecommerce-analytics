"""Build data/olist.duckdb (and optionally load Postgres) from data/raw."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from olist.config import RAW_DIR, WAREHOUSE_PATH
from olist.db import build_duckdb, build_postgres


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW_DIR)
    parser.add_argument("--duckdb", type=Path, default=WAREHOUSE_PATH)
    parser.add_argument("--postgres", action="store_true", help="also load OLIST_PG_DSN")
    args = parser.parse_args()

    t0 = time.time()
    build_duckdb(args.duckdb, args.raw)
    print(f"duckdb warehouse at {args.duckdb} ({time.time() - t0:.1f}s)")
    if args.postgres:
        t0 = time.time()
        build_postgres(os.environ["OLIST_PG_DSN"], args.raw)
        print(f"postgres loaded ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()

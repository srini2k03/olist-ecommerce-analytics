"""Rebuild app/data from the DuckDB warehouse (run after build_warehouse.py)."""

from __future__ import annotations

import argparse
import time

from olist.app_data import build


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-models", action="store_true", help="only the order/item bundle and summaries"
    )
    args = parser.parse_args()
    t0 = time.time()
    sizes = build(with_models=not args.skip_models)
    for name, size in sizes.items():
        print(f"{name:<28} {size / 1024:8.1f} KiB")
    print(f"total {sum(sizes.values()) / 1024 / 1024:.2f} MiB in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

"""Rebuild app/data from the DuckDB warehouse (run after build_warehouse.py)."""

from __future__ import annotations

import time

from olist.app_data import build


def main() -> None:
    t0 = time.time()
    sizes = build()
    for name, size in sizes.items():
        print(f"{name:<28} {size / 1024:8.1f} KiB")
    print(f"total {sum(sizes.values()) / 1024 / 1024:.2f} MiB in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

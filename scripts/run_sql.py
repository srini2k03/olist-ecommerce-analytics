"""Run every query in sql/, snapshot DuckDB results and optionally cross-check Postgres.

Examples:
    python scripts/run_sql.py                      # DuckDB, write results/sql/*.csv
    python scripts/run_sql.py --postgres           # also run on OLIST_PG_DSN and compare
    python scripts/run_sql.py --no-snapshot --postgres
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from olist.config import SQL_RESULTS_DIR, WAREHOUSE_PATH
from olist.db import DuckDBEngine, connect
from olist.sql_runner import frames_match, list_queries, run, snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--duckdb", type=Path, default=WAREHOUSE_PATH)
    parser.add_argument("--postgres", action="store_true", help="compare with OLIST_PG_DSN")
    parser.add_argument("--no-snapshot", action="store_true")
    parser.add_argument("--out", type=Path, default=SQL_RESULTS_DIR)
    args = parser.parse_args()

    duck = DuckDBEngine(args.duckdb)
    pg = connect("postgres") if args.postgres else None
    failures = 0
    for q in list_queries():
        t0 = time.time()
        res = run(duck, q)
        line = f"{q.name:<34} duckdb {len(res):>4} rows {time.time() - t0:5.2f}s"
        if not args.no_snapshot:
            snapshot(res, q.name, args.out)
        if pg is not None:
            t0 = time.time()
            ok, msg = frames_match(res, run(pg, q))
            line += f" | postgres {time.time() - t0:5.2f}s {'MATCH' if ok else 'DIFF: ' + msg}"
            failures += not ok
        print(line)
    duck.close()
    if pg is not None:
        pg.close()
    if failures:
        print(f"{failures} queries differ between engines", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

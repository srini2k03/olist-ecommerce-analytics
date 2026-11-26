"""Load the marts into BigQuery (sandbox) and check all analytics queries against DuckDB.

Usage (in CI, after ``python -m olist.pipeline`` and Google auth):
    python scripts/bigquery_parity.py --project PROJECT_ID [--dataset olist] [--location US]

Writes results/bigquery/parity.json and parity.md and exits non-zero if any query differs.
BigQuery sandbox tables expire after 60 days, so the workflow reloads them monthly.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from olist.bigquery import MART_TABLES, BigQueryEngine, bigquery_sql, export_marts
from olist.config import REPO_ROOT
from olist.db import DuckDBEngine
from olist.sql_runner import frames_match, list_queries, normalise

OUT = REPO_ROOT / "results" / "bigquery"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--dataset", default="olist")
    ap.add_argument("--location", default="US")
    args = ap.parse_args()

    duck = DuckDBEngine()
    bq = BigQueryEngine(project=args.project, dataset=args.dataset, location=args.location)
    bq.ensure_dataset()

    loads = {}
    with tempfile.TemporaryDirectory() as tmp:
        for table, path in export_marts(duck, Path(tmp)).items():
            start = time.perf_counter()
            rows = bq.load_parquet(table, path)
            loads[table] = {"rows": rows, "seconds": round(time.perf_counter() - start, 2)}
            print(f"loaded {table}: {rows} rows", flush=True)

    queries, failures = [], 0
    for q in list_queries():
        sql, how = bigquery_sql(q)
        start = time.perf_counter()
        expected = normalise(duck.query(q.sql))
        duck_ms = (time.perf_counter() - start) * 1000
        try:
            got = normalise(bq.query(sql))
            ok, why = frames_match(expected, got)
        except Exception as e:  # report every query, do not stop at the first error
            ok, why = False, f"BigQuery error: {e}"[:500]
        failures += not ok
        queries.append(
            {
                "query": q.name,
                "domain": q.domain,
                "bigquery_sql": how,
                "match": ok,
                "detail": why,
                "rows": int(len(expected)),
                "duckdb_ms": round(duck_ms, 1),
                "bigquery_ms": round(bq.last_job.elapsed_ms, 1),
                "bigquery_bytes_processed": bq.last_job.bytes_processed,
            }
        )
        print(f"{'MATCH' if ok else 'DIFF '} {q.name} ({how}) {'' if ok else why}", flush=True)

    commit = os.environ.get("GITHUB_SHA", "local")  # set by GitHub Actions
    report = {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "git_commit": commit,
        "project_dataset": f"{args.dataset} (location {args.location}, BigQuery sandbox)",
        "tables_loaded": loads,
        "queries": queries,
        "matched": len(queries) - failures,
        "total": len(queries),
        "bytes_processed_total": sum(q["bigquery_bytes_processed"] for q in queries),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "parity.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# BigQuery parity",
        "",
        f"- Generated {report['generated_at']} at commit `{commit[:10]}`, "
        "BigQuery sandbox (no billing).",
        f"- **{report['matched']} / {report['total']} analytics queries return the same result "
        "on BigQuery as on DuckDB** (cell by cell, relative tolerance 1e-6).",
        f"- {len(MART_TABLES)} marts loaded from Parquet: "
        + ", ".join(f"{t} ({v['rows']:,} rows)" for t, v in loads.items())
        + ".",
        f"- Total bytes processed by the queries: {report['bytes_processed_total'] / 1e6:,.1f} MB "
        "(the sandbox allows 1 TB a month).",
        "",
        "| Query | BigQuery SQL | Rows | Match | DuckDB ms | BigQuery ms | MB processed |",
        "|---|---|---|---|---|---|---|",
    ]
    for q in queries:
        match = "yes" if q["match"] else "NO: " + q["detail"][:80]
        mb = q["bigquery_bytes_processed"] / 1e6
        lines.append(
            f"| {q['query']} | {q['bigquery_sql']} | {q['rows']} | {match} "
            f"| {q['duckdb_ms']:.0f} | {q['bigquery_ms']:.0f} | {mb:.1f} |"
        )
    (OUT / "parity.md").write_text("\n".join(lines) + "\n")
    print(f"{report['matched']}/{report['total']} queries match")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

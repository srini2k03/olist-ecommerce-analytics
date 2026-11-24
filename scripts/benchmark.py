"""Benchmark the pipeline, the SQL library, the metric layer and the dashboard computations.

    python scripts/benchmark.py [--out results/benchmarks/benchmark.json] [--repeats 10]
                                [--postgres] [--postgres-regressions]

Each stage runs in a fresh subprocess so its peak memory (max RSS) is its own. Query
latencies are medians over ``--repeats`` runs after one warm-up. With --postgres (needs
OLIST_PG_DSN) the same layers and queries are timed on Postgres; --postgres-regressions also
times the two slow variants fixed in 492a77f (NOT EXISTS checks, marts without ANALYZE),
each capped by a statement timeout.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

STAGES = {
    "ingest_raw": """
from olist.db import build_duckdb
build_duckdb(DB)
""",
    "checks_raw": """
from olist.db import DuckDBEngine
from olist.pipeline.checks import run_checks
e = DuckDBEngine(DB); run_checks(e, "raw")
""",
    "build_staging_marts": """
from olist.db import DuckDBEngine
from olist.pipeline.models import build_models
e = DuckDBEngine(DB, read_only=False); STATS.update(build_models(e)); e.close()
""",
    "checks_staging_marts": """
from olist.db import DuckDBEngine
from olist.pipeline.checks import run_checks
e = DuckDBEngine(DB); run_checks(e, "staging"); run_checks(e, "marts")
""",
    "metric_layer_full_window": """
import pandas as pd
from olist.db import DuckDBEngine
from olist import metrics
from olist.config import METRICS_SPEC
e = DuckDBEngine(DB)
o = e.query("SELECT * FROM fct_orders"); i = e.query("SELECT * FROM fct_order_items")
spec = metrics.load_spec(METRICS_SPEC)
t = time.perf_counter(); metrics.compute_all(o, i, metrics.assumption_values(spec))
STATS["compute_all_s"] = time.perf_counter() - t
""",
}
TEMPLATE = """
import json, resource, sys, time
sys.path[:0] = [{src!r}, {app!r}]
DB = {db!r}
STATS = {{}}
t0 = time.perf_counter()
{body}
STATS["seconds"] = time.perf_counter() - t0
STATS["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
print("BENCH" + json.dumps(STATS, default=float))
"""


def run_stage(name: str, body: str, db: Path) -> dict:
    code = TEMPLATE.format(src=str(ROOT / "src"), app=str(ROOT / "app"), db=str(db), body=body)
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=ROOT
    )
    line = next(ln for ln in out.stdout.splitlines() if ln.startswith("BENCH"))
    stats = json.loads(line[5:])
    print(f"{name:<28} {stats['seconds']:7.2f} s  peak {stats['peak_rss_mb']:7.1f} MB")
    return stats


def query_latency(engine, repeats: int) -> dict:
    import statistics

    from olist.sql_runner import list_queries

    out = {}
    for q in list_queries():
        engine.query(q.sql)  # warm-up
        times = []
        for _ in range(repeats):
            t = time.perf_counter()
            engine.query(q.sql)
            times.append(time.perf_counter() - t)
        times.sort()
        out[q.name] = {
            "median_ms": round(statistics.median(times) * 1000, 2),
            "p95_ms": round(times[min(len(times) - 1, int(0.95 * len(times)))] * 1000, 2),
        }
    return out


def dashboard_compute(repeats: int) -> dict:
    """Server-side cost of what the dashboard does on each rerun (same code as the browser)."""
    import statistics

    from dashboard import data

    from olist.analytics import cohorts, rfm, sellers
    from olist.metrics import assumption_values, compute_all

    t = time.perf_counter()
    d = data.load()
    load_s = time.perf_counter() - t
    assume = assumption_values(d["metrics_spec"])
    top_cat = d["items"].groupby("category", observed=True).size().idxmax()
    top_seller = int(d["items"].groupby("seller_id").size().idxmax())
    cases = {
        "default (2017-01..2018-08)": data.Filters(),
        "customer state SP": data.Filters(customer_states=("SP",)),
        f"category {top_cat}": data.Filters(categories=(str(top_cat),)),
        "one seller": data.Filters(sellers=(top_seller,)),
    }
    out = {"bundle_load_s": round(load_s, 3), "cases": {}}
    for label, f in cases.items():
        rows = {}
        ts = []
        for _ in range(repeats):
            data.apply.clear()  # time the filtering itself, not the cache
            t = time.perf_counter()
            res = data.apply(f)
            ts.append(time.perf_counter() - t)
        rows["filter_ms"] = round(statistics.median(ts) * 1000, 1)
        o, it = res
        for step, fn in {
            "metrics": lambda o=o, it=it: compute_all(o, it, assume),
            "cohort_matrix": lambda o=o: cohorts.retention_matrix(o[o.is_valid == 1]),
            "rfm": lambda o=o: rfm.segment_summary(rfm.score(rfm.rfm_table(o, "2018-09-01"))),
            "seller_ranking": lambda o=o, it=it: sellers.rank_sellers(
                sellers.scorecard_from_frames(o, it), n_draws=200
            ),
        }.items():
            ts = []
            for _ in range(max(3, repeats // 3)):
                t = time.perf_counter()
                fn()
                ts.append(time.perf_counter() - t)
            rows[f"{step}_ms"] = round(statistics.median(ts) * 1000, 1)
        rows["orders"] = len(o)
        out["cases"][label] = rows
        print(f"dashboard {label:<34} {rows}")
    return out


def postgres(dsn: str, repeats: int, regressions: bool) -> dict:
    from olist.db import PostgresEngine, build_postgres
    from olist.pipeline.checks import run_checks
    from olist.pipeline.models import build_models

    res: dict = {}
    t = time.perf_counter()
    build_postgres(dsn)
    res["ingest_raw_s"] = round(time.perf_counter() - t, 2)
    eng = PostgresEngine(dsn)
    t = time.perf_counter()
    run_checks(eng, "raw")
    res["checks_raw_s"] = round(time.perf_counter() - t, 2)
    t = time.perf_counter()
    models = build_models(eng)
    res["build_staging_marts_s"] = round(time.perf_counter() - t, 2)
    res["fct_orders_s"] = models["fct_orders"]["seconds"]
    t = time.perf_counter()
    run_checks(eng, "staging")
    run_checks(eng, "marts")
    res["checks_staging_marts_s"] = round(time.perf_counter() - t, 2)
    res["queries"] = query_latency(eng, repeats)
    if regressions:
        res["regressions"] = postgres_regressions(eng)
    eng.close()
    return res


def postgres_regressions(eng) -> dict:
    """Time the two slow variants fixed in 492a77f, capped at 10 minutes each."""
    from olist.pipeline.models import build_order, list_models

    out = {}
    eng.execute("SET statement_timeout = '600s'")
    old_fk = (
        "SELECT SUM(CASE WHEN c.customer_id IS NOT NULL AND NOT EXISTS (SELECT 1 FROM customers "
        "AS p WHERE p.customer_id = c.customer_id) THEN 1 ELSE 0 END) AS failing FROM orders AS c"
    )
    new_fk = (
        "SELECT SUM(CASE WHEN c.customer_id IS NOT NULL AND p.k IS NULL THEN 1 ELSE 0 END) "
        "AS failing FROM orders AS c LEFT JOIN (SELECT DISTINCT customer_id AS k FROM customers) "
        "AS p ON p.k = c.customer_id"
    )
    for label, sql in (("fk_check_not_exists_in_case", old_fk), ("fk_check_anti_join", new_fk)):
        t = time.perf_counter()
        try:
            eng.query(sql)
            out[label + "_s"] = round(time.perf_counter() - t, 2)
        except Exception as exc:  # statement timeout
            out[label + "_s"] = f"> 600 (cancelled: {type(exc).__name__})"
    # fct_orders on staging tables without planner statistics vs with them
    staging = [m for m in build_order(list_models()) if m.layer == "staging"]
    fct = next(m for m in list_models() if m.name == "fct_orders")
    for m in staging:  # recreate without ANALYZE: fresh tables have no statistics
        eng.execute(f"DROP TABLE IF EXISTS {m.name} CASCADE")
        eng.execute(f"CREATE TABLE {m.name} AS\n{m.select}")
    for label, analyze in (
        ("fct_orders_without_analyze", False),
        ("fct_orders_with_analyze", True),
    ):
        if analyze:
            for m in staging:
                eng.execute(f"ANALYZE {m.name}")
        eng.execute("DROP TABLE IF EXISTS fct_orders_bench")
        t = time.perf_counter()
        try:
            eng.execute(f"CREATE TABLE fct_orders_bench AS\n{fct.select}")
            out[label + "_s"] = round(time.perf_counter() - t, 2)
        except Exception as exc:
            out[label + "_s"] = f"> 600 (cancelled: {type(exc).__name__})"
    eng.execute("DROP TABLE IF EXISTS fct_orders_bench")
    eng.execute("SET statement_timeout = 0")
    return out


def environment() -> dict:
    from olist.provenance import versions

    git = ["git", "rev-parse", "--short", "HEAD"]
    sha = subprocess.run(git, capture_output=True, text=True, cwd=ROOT).stdout.strip()  # noqa: S603,S607
    return {
        "date_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M"),
        "git_sha": sha,
        "machine": os.environ.get("BENCH_MACHINE", platform.node() or "unknown"),
        "platform": platform.platform(),
        "cpus": os.cpu_count(),
        "versions": versions(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out", type=Path, default=ROOT / "results" / "benchmarks" / "benchmark.json"
    )
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "bench.duckdb")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--postgres", action="store_true")
    parser.add_argument("--postgres-regressions", action="store_true")
    args = parser.parse_args()

    result = {"environment": environment(), "duckdb": {}}
    for name, body in STAGES.items():
        result["duckdb"][name] = run_stage(name, body, args.db)
    from olist.db import DuckDBEngine

    eng = DuckDBEngine(args.db)
    result["duckdb"]["queries"] = query_latency(eng, args.repeats)
    eng.close()
    args.db.unlink(missing_ok=True)
    slow = sorted(result["duckdb"]["queries"].items(), key=lambda kv: -kv[1]["median_ms"])[:3]
    print("slowest queries:", [(k, v["median_ms"]) for k, v in slow])
    result["dashboard"] = dashboard_compute(args.repeats)
    bundle = ROOT / "app" / "data"
    result["dashboard"]["bundle_mb"] = round(
        sum(p.stat().st_size for p in bundle.iterdir()) / 1024 / 1024, 2
    )
    if args.postgres:
        dsn = os.environ.get("OLIST_PG_DSN")
        if not dsn:
            parser.error("--postgres needs OLIST_PG_DSN")
        result["postgres"] = postgres(dsn, args.repeats, args.postgres_regressions)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

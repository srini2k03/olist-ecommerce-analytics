"""Command line entry point: ``python -m recsys <command>``.

Commands (real data, run in this order)::

    python -m recsys fetch                     # Kaggle download -> data/hm/raw, parquet
    python -m recsys subset --weeks 16 --frac 0.25
    python -m recsys split
    python -m recsys baselines --split validation
    python -m recsys retrieval                 # two-tower + FAISS
    python -m recsys all                       # every modelling stage after split

``python -m recsys --synthetic`` runs the same pipeline end to end on generated
data under ``data/hm/synthetic/`` (results included), for tests and CI. Its
numbers describe the generator, not H&M, and never go to ``results/recsys/``.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from recsys import config
from recsys.baselines import (
    AgePopularityRecommender,
    ItemCooccurrenceRecommender,
    PopularityRecommender,
    Recommender,
    RepurchaseRecommender,
)
from recsys.context import SPLIT_NAMES, SYNTHETIC_MARKER, Context, make_context, records
from recsys.data.schema import HMData, write_json
from recsys.data.split import check_strictly_before, cold_start_report, split_transactions
from recsys.data.subset import make_subset
from recsys.evaluate.metrics import evaluate

log = logging.getLogger("recsys")

SUMMARY_COLUMNS = [
    ("map", 12),
    ("recall", 12),
    ("ndcg", 12),
    ("hit_rate", 12),
    ("recall", 100),
    ("coverage", 12),
    ("popularity_pct", 12),
]


# ----------------------------------------------------------------------------- data steps


def run_fetch(data_dir: Path, raw_dir: Path, force: bool = False, skip_download: bool = False):
    from recsys.data import fetch_hm

    if not skip_download:
        fetch_hm.download(raw_dir, force=force)
    return fetch_hm.convert(raw_dir, data_dir / config.PARQUET_SUBDIR)


def run_subset(data_dir: Path, weeks: int, frac: float, seed: int) -> dict:
    full = HMData.read(data_dir / config.PARQUET_SUBDIR)
    sub = make_subset(full, n_weeks=weeks, customer_frac=frac, seed=seed)
    out = data_dir / config.SUBSET_SUBDIR
    sub.write(out)
    tx = sub.transactions
    manifest = {
        "params": {"weeks": weeks, "customer_frac": frac, "seed": seed},
        "full": full.counts(),
        "subset": sub.counts(),
        "date_range": [str(tx.t_dat.min().date()), str(tx.t_dat.max().date())],
    }
    write_json(out / "manifest.json", manifest)
    return manifest


def run_split(data_dir: Path, val_weeks: int, test_weeks: int, train_weeks: int | None) -> dict:
    tx = HMData.read(data_dir / config.SUBSET_SUBDIR).transactions
    split = split_transactions(tx, val_weeks, test_weeks, train_weeks)
    out = data_dir / config.SPLIT_SUBDIR
    out.mkdir(parents=True, exist_ok=True)
    for name in SPLIT_NAMES:
        getattr(split, name).to_parquet(out / f"{name}.parquet", index=False)
    report = {
        "params": {"val_weeks": val_weeks, "test_weeks": test_weeks, "train_weeks": train_weeks},
        "boundaries": split.boundaries.to_dict(),
        "rows": {name: len(getattr(split, name)) for name in SPLIT_NAMES},
        "leakage_checks": "passed",
        "cold_start": {
            name: cold_start_report(split.history_for(name), split.target(name))
            for name in ("validation", "test")
        },
    }
    write_json(out / "split_report.json", report)
    for name, cs in report["cold_start"].items():
        log.info(
            "%s: %d users (%d cold, %.1f%%), %d items (%d cold, %.1f%%)",
            name,
            cs["target_users"],
            cs["cold_users"],
            100 * cs["cold_user_share"],
            cs["target_items"],
            cs["cold_items"],
            100 * cs["cold_item_share"],
        )
    return report


# ----------------------------------------------------------------------------- modelling stages


def make_baselines(customers: pd.DataFrame) -> list[Recommender]:
    return [
        PopularityRecommender(weeks=1),
        AgePopularityRecommender(customers, weeks=4),
        ItemCooccurrenceRecommender(),
        RepurchaseRecommender(),
    ]


def baseline_params(model: Recommender) -> dict:
    return {k: v for k, v in vars(model).items() if not k.endswith("_") and k != "customers"}


def run_baselines(
    ctx: Context, target: str = "validation", ks: tuple[int, ...] = config.EVAL_KS
) -> pd.DataFrame:
    split = ctx.split
    history, truth_rows = split.history_for(target), split.target(target)
    check_strictly_before(history, split.boundaries.window(target)[0], f"{target} history")
    fold = ctx.folds[target]
    truth, users = fold.truth, fold.users
    item_counts = history.groupby("article_id").size()

    tables, params = [], {}
    for model in make_baselines(ctx.data.customers):
        recs = model.fit(history).recommend(users, max(ks))
        ctx.save_recs(target, model.name, recs)
        table = evaluate(recs, truth, ks, item_counts=item_counts, n_boot=ctx.n_boot, seed=ctx.seed)
        table.insert(0, "model", model.name)
        tables.append(table)
        params[model.name] = baseline_params(model)
        map12 = table.query("metric == 'map' and k == 12").value.iloc[0]
        log.info("%s: MAP@12 = %.4f", model.name, map12)
    result = pd.concat(tables, ignore_index=True)
    ctx.write_result(
        f"baselines_{target}",
        {
            "split": target,
            "history_rows": len(history),
            "eval_users": len(users),
            "cold_start": cold_start_report(history, truth_rows),
            "baselines": params,
            "metrics": records(result),
        },
    )
    return result


def _retrieval(ctx: Context) -> pd.DataFrame:
    from recsys.retrieval.run import run_retrieval

    return run_retrieval(ctx)


def _ranking(ctx: Context) -> pd.DataFrame:
    from recsys.ranking.run import run_ranking

    return run_ranking(ctx)


def _coldstart(ctx: Context) -> pd.DataFrame:
    from recsys.coldstart.run import run_coldstart

    return run_coldstart(ctx)


def _report(ctx: Context) -> dict:
    from recsys.evaluate.report import run_report

    return run_report(ctx)


STAGES: dict[str, Callable[[Context], pd.DataFrame | dict]] = {
    "baselines": lambda ctx: run_baselines(ctx, "validation"),
    "baselines_test": lambda ctx: run_baselines(ctx, "test"),
    "retrieval": _retrieval,
    "ranking": _ranking,
    "coldstart": _coldstart,
    "report": _report,
}


def run_all(ctx: Context) -> None:
    for name, stage in STAGES.items():
        log.info("=== stage %s ===", name)
        stage(ctx)


def summary_table(result: pd.DataFrame) -> pd.DataFrame:
    wide = result.pivot_table(index="model", columns=["metric", "k"], values="value", sort=False)
    cols = [c for c in SUMMARY_COLUMNS if c in wide.columns]
    out = wide[cols]
    out.columns = [f"{m}@{k}" for m, k in cols]
    return out.sort_values(out.columns[0], ascending=False)


def run_synthetic(
    data_dir: Path, results_dir: Path | None, seed: int, n_boot: int, quick: bool
) -> Context:
    from recsys.synthetic import generate

    log.info("synthetic run in %s (numbers describe the generator, not H&M)", data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / SYNTHETIC_MARKER).write_text("Generated by recsys.synthetic; not H&M data.\n")
    generate(seed=seed).write(data_dir / config.PARQUET_SUBDIR)
    run_subset(data_dir, weeks=config.SUBSET_WEEKS, frac=1.0, seed=seed)
    run_split(data_dir, val_weeks=1, test_weeks=1, train_weeks=None)
    ctx = make_context(data_dir, results_dir, seed, n_boot, quick)
    run_all(ctx)
    return ctx


# ----------------------------------------------------------------------------- argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m recsys", description=__doc__.split("\n")[0])
    p.add_argument("--synthetic", action="store_true", help="run every stage on generated data")
    p.add_argument(
        "--data-dir", type=Path, help="default: data/hm (data/hm/synthetic if synthetic)"
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        help="default: results/recsys for real data, <data-dir>/results for synthetic data",
    )
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    p.add_argument("--n-boot", type=int, default=1000, help="bootstrap resamples for CIs")
    p.add_argument("--quick", action="store_true", help="small models and grids (smoke tests)")
    sub = p.add_subparsers(dest="command")

    f = sub.add_parser("fetch", help="download the 3 CSVs with the Kaggle CLI, convert to parquet")
    f.add_argument("--raw-dir", type=Path, default=config.HM_RAW_DIR)
    f.add_argument("--force", action="store_true", help="re-download even if files exist")
    f.add_argument("--skip-download", action="store_true", help="only convert existing CSVs")

    s = sub.add_parser("subset", help="last N weeks and a sample of customers")
    s.add_argument("--weeks", type=int, default=config.SUBSET_WEEKS)
    s.add_argument("--frac", type=float, default=config.SUBSET_CUSTOMER_FRAC)

    sp = sub.add_parser("split", help="week-based train/validation/test split with leakage checks")
    sp.add_argument("--val-weeks", type=int, default=1)
    sp.add_argument("--test-weeks", type=int, default=1)
    sp.add_argument("--train-weeks", type=int, default=None, help="default: all earlier weeks")

    b = sub.add_parser("baselines", help="fit and evaluate every baseline")
    b.add_argument("--split", choices=["validation", "test"], default="validation")
    for name in STAGES:
        if not name.startswith("baselines"):
            sub.add_parser(name, help=f"run the {name} stage")
    sub.add_parser("all", help="run every modelling stage (after split)")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("faiss").setLevel(logging.WARNING)
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.synthetic:
        if args.command:
            raise SystemExit("--synthetic runs the whole pipeline; do not pass a command")
        data_dir = args.data_dir or config.HM_DIR / "synthetic"
        run_synthetic(data_dir, args.results_dir, args.seed, args.n_boot, args.quick)
        return 0
    if not args.command:
        parser.print_help()
        return 2
    data_dir = args.data_dir or config.HM_DIR
    if args.command == "fetch":
        from recsys.data.fetch_hm import MissingCredentialsError

        try:
            run_fetch(data_dir, args.raw_dir, args.force, args.skip_download)
        except MissingCredentialsError as exc:
            log.error("%s", exc)
            return 1
        return 0
    if args.command == "subset":
        run_subset(data_dir, args.weeks, args.frac, args.seed)
        return 0
    if args.command == "split":
        run_split(data_dir, args.val_weeks, args.test_weeks, args.train_weeks)
        return 0
    ctx = make_context(data_dir, args.results_dir, args.seed, args.n_boot, args.quick)
    if args.command == "baselines":
        print(summary_table(run_baselines(ctx, args.split)).round(4).to_string())
    elif args.command == "all":
        run_all(ctx)
    else:
        result = STAGES[args.command](ctx)
        if isinstance(result, pd.DataFrame) and "model" in result:
            print(summary_table(result).round(4).to_string())
    return 0

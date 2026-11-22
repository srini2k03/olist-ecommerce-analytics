"""Phase 5: one test-week report comparing every stage, plus the simulated A/B test.

Reads the test recommendations each stage saved under ``artifacts/recs/test/``
and the stage result files, and writes ``report.json`` and ``report.md`` to the
results directory (the gitignored synthetic results dir for synthetic data).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from recsys.abtest import run_ab_test
from recsys.context import Context, records
from recsys.evaluate.metrics import evaluate, paired_bootstrap_diff, per_user_metrics

log = logging.getLogger(__name__)

BASELINES = ["popularity_1w", "age_popularity_4w", "item_cooccurrence", "repurchase"]
MODELS = ["two_tower", "two_tower_ranker", "ranker_with_cold_start"]
STAGE_LABELS = {
    "popularity_1w": "Popularity (last week)",
    "age_popularity_4w": "Popularity by age bucket",
    "item_cooccurrence": "Item co-occurrence",
    "repurchase": "Repurchase",
    "two_tower": "Two-tower retrieval",
    "two_tower_ranker": "Two-tower + LambdaRank",
    "ranker_with_cold_start": "Two-tower + LambdaRank + cold start",
}
TABLE_COLUMNS = [
    ("recall", 12),
    ("recall", 50),
    ("ndcg", 12),
    ("map", 12),
    ("hit_rate", 12),
    ("coverage", 12),
    ("popularity_pct", 12),
    ("rec_gini", 12),
]
DIFF_METRICS = ("map", "recall", "hit_rate")


def _load(ctx: Context, name: str) -> dict | None:
    path = ctx.results_dir / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def best_baseline(ctx: Context, test_table: pd.DataFrame) -> tuple[str, str]:
    """The baseline with the best validation MAP@12 (test MAP@12 if validation is missing)."""
    val = _load(ctx, "baselines_validation")
    if val:
        rows = [r for r in val["metrics"] if r["metric"] == "map" and r["k"] == 12]
        rows = [r for r in rows if r["model"] in BASELINES]
        if rows:
            return max(rows, key=lambda r: r["value"])["model"], "validation MAP@12"
    sub = test_table.query("metric == 'map' and k == 12 and model in @BASELINES")
    return sub.loc[sub.value.idxmax(), "model"], "test MAP@12"


def run_report(ctx: Context, shrinkage: float = 0.5) -> dict:
    fold = ctx.folds["test"]
    truth = fold.truth
    item_counts = fold.history.groupby("article_id").size()
    saved = ctx.load_recs("test")
    stages = [s for s in BASELINES + MODELS if s in saved]
    missing = [s for s in BASELINES if s not in saved]
    if missing:
        raise RuntimeError(f"run `baselines --split test` first; missing {missing}")

    tables, per_user = [], {}
    for stage in stages:
        t = evaluate(saved[stage], truth, (12, 50), item_counts, n_boot=ctx.n_boot, seed=ctx.seed)
        tables.append(t.assign(model=stage))
        per_user[stage] = per_user_metrics(saved[stage], truth, 12)
    table = pd.concat(tables, ignore_index=True)
    base, base_rule = best_baseline(ctx, table)

    diffs = []
    for stage in stages:
        if stage == base:
            continue
        for metric in DIFF_METRICS:
            a = per_user[base][metric].to_numpy()
            b = per_user[stage][metric].reindex(per_user[base].index).to_numpy()
            d = paired_bootstrap_diff(a, b, ctx.n_boot, seed=ctx.seed)
            diffs.append({"model": stage, "metric": f"{metric}@12", "vs": base, **d})

    treatment = next((m for m in reversed(MODELS) if m in saved), None)
    ab = None
    if treatment:
        ab = run_ab_test(
            per_user[base]["hit_rate"].to_numpy(),
            per_user[treatment]["hit_rate"].reindex(per_user[base].index).to_numpy(),
            base,
            treatment,
            shrinkage=shrinkage,
            seed=ctx.seed,
            offline_diff=next(
                (d for d in diffs if d["model"] == treatment and d["metric"] == "hit_rate@12"),
                None,
            ),
        )
        log.info("A/B decision: %s", ab["decision"])

    report = {
        "split": "test",
        "eval_users": len(truth),
        "best_baseline": {"model": base, "chosen_by": base_rule},
        "stages": records(table),
        "paired_diffs": diffs,
        "ab_test": ab,
        "retrieval": _load(ctx, "retrieval_validation"),
        "ranking": _load(ctx, "ranking"),
        "coldstart": _load(ctx, "coldstart"),
    }
    path = ctx.write_result("report", report)
    full = json.loads(path.read_text())
    md = render_markdown(full)
    (ctx.results_dir / "report.md").write_text(md)
    return full


# ----------------------------------------------------------------------------- markdown


def _fmt(v, digits: int = 4) -> str:
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def stage_table(report: dict) -> str:
    rows = pd.DataFrame(report["stages"])
    head = ["Stage", *[f"{m}@{k}" for m, k in TABLE_COLUMNS]]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for stage, g in rows.groupby("model", sort=False):
        cells = [STAGE_LABELS.get(stage, stage)]
        for m, k in TABLE_COLUMNS:
            r = g[(g.metric == m) & (g.k == k)]
            if r.empty:
                cells.append("–")
                continue
            r = r.iloc[0]
            cell = _fmt(r["value"])
            if r["ci_low"] is not None and m in ("recall", "map", "hit_rate", "ndcg"):
                cell += f" ({_fmt(r['ci_low'])}–{_fmt(r['ci_high'])})"
            cells.append(cell)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def diff_table(report: dict) -> str:
    lines = ["| Stage | Metric | Difference vs best baseline (95% CI) |", "|---|---|---|"]
    for d in report["paired_diffs"]:
        lines.append(
            f"| {STAGE_LABELS.get(d['model'], d['model'])} | {d['metric']} | "
            f"{d['diff']:+.4f} ({d['ci_low']:+.4f} to {d['ci_high']:+.4f}) |"
        )
    return "\n".join(lines)


def ablation_table(ranking: dict | None) -> str:
    if not ranking:
        return "_Ranking stage not run._"
    lines = ["| Dropped feature group | Validation MAP@12 | Difference (95% CI) |", "|---|---|---|"]
    for a in ranking["ablation"]:
        if a["dropped"] == "none":
            lines.append(f"| none (all features) | {a['val_map@12']:.4f} | – |")
        else:
            lines.append(
                f"| {a['dropped']} | {a['val_map@12']:.4f} | "
                f"{a['diff']:+.4f} ({a['ci_low']:+.4f} to {a['ci_high']:+.4f}) |"
            )
    top = ", ".join(f"`{r['feature']}` ({r['gain_share']:.0%})" for r in ranking["importance"][:5])
    return "\n".join(lines) + f"\n\nTop LightGBM gain importances: {top}."


def coldstart_section(cold: dict | None) -> str:
    if not cold:
        return "_Cold-start stage not run._"
    cu = cold["cold_users"]
    rows = [r for r in cu["metrics"] if r["k"] == 12 and r["metric"] in ("map", "recall")]
    df = pd.DataFrame(rows).pivot_table(
        index=["segment", "model"], columns="metric", values="value", sort=False
    )
    lines = [
        f"Cold users (no purchase before the test week): {cu['cold_test_users']:,} of "
        f"{cu['test_users']:,} ({cu['cold_share']:.1%}). Blend weight on age popularity chosen on "
        f"validation: {cu['chosen_alpha']:.2f}; cold users are served by "
        f"**{cu['cold_path_validation']['choice']}** (validation MAP@12: blend "
        f"{_fmt(cu['cold_path_validation']['val_map@12_blend'])}, ranker "
        f"{_fmt(cu['cold_path_validation']['val_map@12_ranker'])}).",
        "",
        "| Segment | Strategy | MAP@12 | Recall@12 |",
        "|---|---|---|---|",
    ]
    for (seg, model), r in df.iterrows():
        lines.append(f"| {seg} | {model} | {_fmt(r.get('map'))} | {_fmt(r.get('recall'))} |")
    ni = cold["new_items"]
    lines += [
        "",
        f"New articles (first sold in the test week): {ni['new_articles_bought']:,} articles, "
        f"{ni['new_item_purchases']:,} purchases by {ni['users_buying_new_items']:,} users; "
        f"{ni['unseen_catalogue_articles']:,} catalogue articles had no sales before the "
        "test week.",
        "",
        "| Retrieval variant | Index size | New-article recall@50 | New-article share@50 "
        "| Recall@12 (all users) |",
        "|---|---|---|---|---|",
    ]
    for v in ni["variants"]:
        lines.append(
            f"| {v['model']} | {v['index_size']:,} | {_fmt(v['new_item_recall@50'])} | "
            f"{_fmt(v['new_item_share@50'])} | {_fmt(v['recall@12_all_users'])} |"
        )
    return "\n".join(lines)


def retrieval_section(ret: dict | None) -> str:
    if not ret:
        return "_Retrieval stage not run._"
    ann = ret["ann"]
    rows = [r for r in ret["metrics"] if r["metric"] == "recall" and r["k"] in (100, 200)]
    df = pd.DataFrame(rows).pivot_table(index="model", columns="k", values="value", sort=False)
    lines = [
        "| Candidate generator (validation) | Recall@100 | Recall@200 |",
        "|---|---|---|",
        *[f"| {m} | {r[100]:.4f} | {r[200]:.4f} |" for m, r in df.iterrows()],
        "",
        f"HNSW vs exact search over {ann['index_items']:,} articles: overlap@100 "
        f"{ann['overlap_with_exact@100']:.3f}, candidate recall@100 loss "
        f"{ann['candidate_recall@100_loss']:+.4f}; single-query p95 latency "
        f"{ann['latency_ms_hnsw']['p95']:.3f} ms (HNSW) vs "
        f"{ann['latency_ms_flat']['p95']:.3f} ms (exact).",
    ]
    return "\n".join(lines)


def ab_section(ab: dict | None) -> str:
    if not ab:
        return "_No model stage to test._"
    lines = [f"**Decision: {ab['decision']}** ({ab['treatment']} vs {ab['control']}).", ""]
    lines += [f"- {r}" for r in ab["reasoning"]]
    return "\n".join(lines)


def render_markdown(report: dict) -> str:
    synthetic = report["data"] == "synthetic"
    banner = (
        "> **SYNTHETIC DATA.** These numbers come from the test generator in "
        "`recsys/synthetic.py`. They check that the pipeline runs; they are not H&M results.\n"
        if synthetic
        else "> Real H&M data (subset described in `data/hm/subset/manifest.json`).\n"
    )
    b = report["boundaries"]
    return "\n".join(
        [
            "# Recommender evaluation report",
            "",
            banner,
            f"Test week {b['test_start']} to {b['test_end']} (exclusive), {report['eval_users']:,} "
            f"users with a purchase. 95% bootstrap CIs over users ({report['bootstrap']['n_boot']} "
            f"resamples). Generated {report['generated_utc']}.",
            "",
            "## All stages on the test week",
            "",
            stage_table(report),
            "",
            f"Best baseline: **{STAGE_LABELS.get(report['best_baseline']['model'])}** "
            f"(chosen by {report['best_baseline']['chosen_by']}).",
            "",
            "## Paired differences vs the best baseline",
            "",
            diff_table(report),
            "",
            "## Candidate retrieval",
            "",
            retrieval_section(report.get("retrieval")),
            "",
            "## Ranker feature ablation",
            "",
            ablation_table(report.get("ranking")),
            "",
            "## Cold start",
            "",
            coldstart_section(report.get("coldstart")),
            "",
            "## Simulated A/B test",
            "",
            ab_section(report.get("ab_test")),
            "",
        ]
    )


def write_markdown(report_json: Path, out: Path) -> None:
    out.write_text(render_markdown(json.loads(report_json.read_text())))

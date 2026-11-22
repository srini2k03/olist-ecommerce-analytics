"""Phase 3 stage: candidates + features per fold, LambdaRank training, tuning,
test evaluation and feature-group ablation."""

from __future__ import annotations

import itertools
import json
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from recsys.baselines.base import popularity_ranking, recent_window
from recsys.context import Context, records
from recsys.data.folds import Fold
from recsys.evaluate.metrics import (
    bootstrap_mean_ci,
    evaluate,
    paired_bootstrap_diff,
    per_user_metrics,
)
from recsys.ranking.candidates import CandidateConfig, CandidateGenerator
from recsys.ranking.features import FEATURE_GROUPS, FEATURES, FeatureStore, assemble
from recsys.ranking.ranker import LambdaRanker, rank_candidates
from recsys.retrieval.retriever import TwoTowerRetriever
from recsys.retrieval.run import retrieval_config

log = logging.getLogger(__name__)

EVAL_KS = (12, 50)


@dataclass
class FoldData:
    fold: Fold
    retriever: TwoTowerRetriever
    generator: CandidateGenerator
    store: FeatureStore
    frame: pd.DataFrame  # candidates + features + label, sorted by (customer_id, article_id)

    @property
    def fallback(self) -> list[int]:
        """Last-week popularity in the fold's history, to top lists up to k."""
        return popularity_ranking(recent_window(self.fold.history, 1))[:200]


def best_epochs(ctx: Context) -> int:
    meta = ctx.artifacts_dir / "retrieval" / "retriever.json"
    if meta.exists():
        return int(json.loads(meta.read_text())["best_epoch"])
    return retrieval_config(ctx).epochs


def fold_retriever(ctx: Context, fold: Fold, epochs: int) -> TwoTowerRetriever:
    """Validation reuses the phase-2 model (fit on train); other folds are refit on their
    own history with the epoch count phase 2 selected (no early stopping on their target)."""
    data = ctx.data
    saved = ctx.artifacts_dir / "retrieval"
    if fold.name == "validation" and (saved / "retriever.json").exists():
        return TwoTowerRetriever.load(saved, data.articles, data.customers, fold.history)
    cfg = retrieval_config(ctx)
    return TwoTowerRetriever(data.articles, data.customers, cfg).fit(fold.history, epochs=epochs)


def build_fold_data(
    fold: Fold,
    retriever: TwoTowerRetriever,
    articles: pd.DataFrame,
    customers: pd.DataFrame,
    cand_cfg: CandidateConfig | None = None,
) -> FoldData:
    """Everything except the label comes from ``fold.history`` (strictly before the target)."""
    gen = CandidateGenerator(customers, cand_cfg).fit(fold.history, fold.start, retriever)
    cands = gen.generate_offline(fold.users, retriever, fold.history)
    store = FeatureStore.build(fold.history, articles, customers, fold.start)
    frame = assemble(cands, store)
    bought = fold.target[["customer_id", "article_id"]].drop_duplicates().assign(label=1)
    frame = frame.merge(bought, on=["customer_id", "article_id"], how="left")
    frame["label"] = frame["label"].fillna(0).astype(np.int8)
    frame = frame.sort_values(["customer_id", "article_id"], ignore_index=True)
    return FoldData(fold, retriever, gen, store, frame)


def candidate_stats(fd: FoldData, n_boot: int, seed: int) -> dict:
    """How much of the target the candidate union (and each source) covers."""
    truth = fd.fold.truth
    pos = fd.frame[fd.frame["label"] == 1]
    found = pos.groupby("customer_id").size()
    recall = np.array([found.get(u, 0) / len(rel) for u, rel in truth.items()])
    lo, hi = bootstrap_mean_ci(recall, n_boot, seed=seed)
    n_pos = sum(len(r) for r in truth.values())
    by_source = {
        c.removeprefix("src_"): float(pos[c].sum() / n_pos) if n_pos else 0.0
        for c in ("src_two_tower", "src_cooc", "src_repurchase", "src_age_pop")
    }
    return {
        "users": len(truth),
        "candidates": len(fd.frame),
        "candidates_per_user": len(fd.frame) / max(len(truth), 1),
        "positive_rate": float(fd.frame["label"].mean()),
        "candidate_recall": float(recall.mean()),
        "candidate_recall_ci": [float(lo[0]), float(hi[0])],
        "recall_by_source": by_source,
    }


def param_grid(quick: bool) -> list[dict]:
    if quick:
        return [{"num_leaves": 15, "learning_rate": 0.1, "min_data_in_leaf": 20}]
    grid = itertools.product([15, 63], [0.05, 0.1], [20, 100])
    return [{"num_leaves": a, "learning_rate": b, "min_data_in_leaf": c} for a, b, c in grid]


def per_user_ap(recs: dict, truth: dict) -> pd.Series:
    return per_user_metrics(recs, truth, 12)["map"]


def run_ranking(ctx: Context) -> pd.DataFrame:
    data = ctx.data
    epochs = best_epochs(ctx)
    cand_cfg = CandidateConfig(n_two_tower=50) if ctx.quick else CandidateConfig()
    fds = {}
    for name, fold in ctx.folds.items():
        retriever = fold_retriever(ctx, fold, epochs)
        fds[name] = build_fold_data(fold, retriever, data.articles, data.customers, cand_cfg)
        log.info("%s: %d candidates for %d users", name, len(fds[name].frame), len(fold.users))
    train, val, test = fds["ranker_train"], fds["validation"], fds["test"]
    stats = {name: candidate_stats(fd, ctx.n_boot, ctx.seed) for name, fd in fds.items()}

    # ---- light tuning on validation (early stopping on validation NDCG@12)
    rounds = 100 if ctx.quick else 500
    tuning, best = [], None
    for params in param_grid(ctx.quick):
        ranker = LambdaRanker(FEATURES, params, ctx.seed).fit(train.frame, val.frame, rounds)
        recs = rank_candidates(val.frame, ranker.predict(val.frame), 12, val.fallback)
        score = float(per_user_ap(recs, val.fold.truth).mean())
        tuning.append({**params, "best_iteration": ranker.best_iteration_, "val_map@12": score})
        log.info("params %s: validation MAP@12 %.4f", params, score)
        if best is None or score > best[0]:
            best = (score, params, ranker)
    _, best_params, ranker = best
    ranker.save(ctx.artifacts_dir / "ranking" / "ranker.txt")

    # ---- validation and test (once) evaluation of retrieval vs retrieval + ranker
    tables = []
    for fd in (val, test):
        name = fd.fold.name
        truth = fd.fold.truth
        item_counts = fd.fold.history.groupby("article_id").size()
        tt = fd.frame[fd.frame["src_two_tower"] == 1]
        stage_recs = {
            "two_tower": rank_candidates(tt, -tt["tt_rank"].to_numpy(), max(EVAL_KS), fd.fallback),
            "two_tower_ranker": rank_candidates(
                fd.frame, ranker.predict(fd.frame), max(EVAL_KS), fd.fallback
            ),
        }
        for stage, recs in stage_recs.items():
            ctx.save_recs(name, stage, recs)
            t = evaluate(recs, truth, EVAL_KS, item_counts, n_boot=ctx.n_boot, seed=ctx.seed)
            tables.append(t.assign(model=stage, split=name))
    result = pd.concat(tables, ignore_index=True)

    # ---- feature-group ablation on validation (fixed rounds, no early stopping)
    n_iter = ranker.best_iteration_

    def val_ap(features: list[str]) -> pd.Series:
        m = LambdaRanker(features, best_params, ctx.seed).fit(train.frame, None, n_iter, None)
        recs = rank_candidates(val.frame, m.predict(val.frame), 12, val.fallback)
        return per_user_ap(recs, val.fold.truth)

    full = val_ap(FEATURES)
    ablation = [{"dropped": "none", "val_map@12": float(full.mean()), "diff": 0.0}]
    for group, cols in FEATURE_GROUPS.items():
        ap = val_ap([f for f in FEATURES if f not in cols])
        d = paired_bootstrap_diff(
            full.to_numpy(), ap.reindex(full.index).to_numpy(), ctx.n_boot, seed=ctx.seed
        )
        ablation.append({"dropped": group, "val_map@12": float(ap.mean()), **d})
        log.info("without %s: validation MAP@12 %.4f (diff %+.4f)", group, ap.mean(), d["diff"])

    # ---- persist the test-fold state for serving and the report
    folder = ctx.artifacts_dir / "folds" / "test"
    test.retriever.save(folder / "retriever")
    test.generator.save(folder / "candidates.pkl")
    test.store.save(folder / "features")

    ctx.write_result(
        "ranking",
        {
            "two_tower_epochs": epochs,
            "candidates": stats,
            "tuning": tuning,
            "best_params": best_params,
            "best_iteration": n_iter,
            "features": FEATURE_GROUPS,
            "importance": records(ranker.importances()),
            "ablation": ablation,
            "metrics": records(result),
        },
    )
    return result

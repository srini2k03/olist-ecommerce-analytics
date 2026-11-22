"""Phase 2 stage: train the two-tower model, index it, evaluate candidate generation."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from recsys.baselines import (
    AgePopularityRecommender,
    ItemCooccurrenceRecommender,
    PopularityRecommender,
    RepurchaseRecommender,
)
from recsys.context import Context, records
from recsys.data.split import cold_start_report
from recsys.evaluate.metrics import evaluate
from recsys.retrieval import index as faiss_index
from recsys.retrieval.model import TwoTowerConfig
from recsys.retrieval.retriever import TwoTowerRetriever

log = logging.getLogger(__name__)

CANDIDATE_KS = (12, 100, 200)


def retrieval_config(ctx: Context) -> TwoTowerConfig:
    if ctx.quick:
        return TwoTowerConfig(epochs=3, patience=1, batch_size=512, dim=32, seed=ctx.seed)
    return TwoTowerConfig(seed=ctx.seed)


def positions_to_recs(users, positions: np.ndarray, items: np.ndarray) -> dict[int, list[int]]:
    return {
        int(u): [int(items[p]) for p in row if p >= 0]
        for u, row in zip(users, positions, strict=True)
    }


def run_retrieval(ctx: Context, cfg: TwoTowerConfig | None = None) -> pd.DataFrame:
    """Fit on train (early stopping on validation recall@100) and compare exact and
    HNSW retrieval with the phase-1 baselines as candidate generators."""
    fold = ctx.folds["validation"]
    data = ctx.data
    cfg = cfg or retrieval_config(ctx)
    retriever = TwoTowerRetriever(data.articles, data.customers, cfg)
    retriever.fit(fold.history, val_fold=fold)
    out_dir = ctx.artifacts_dir / "retrieval"
    retriever.save(out_dir)

    users, truth = fold.users, fold.truth
    items = retriever.index_items()
    item_vecs = retriever.item_vectors(items)
    user_vecs = retriever.user_vectors(users, fold.history)
    flat = faiss_index.build_flat(item_vecs)
    hnsw = faiss_index.build_hnsw(item_vecs)
    k_max = max(CANDIDATE_KS)
    _, exact_pos = faiss_index.search(flat, user_vecs, k_max)
    _, ann_pos = faiss_index.search(hnsw, user_vecs, k_max)
    faiss_index.save(flat, out_dir / "items_flat.faiss")
    faiss_index.save(hnsw, out_dir / "items_hnsw.faiss")
    np.save(out_dir / "index_items.npy", items)

    item_counts = fold.history.groupby("article_id").size()
    candidates = {
        "two_tower_exact": positions_to_recs(users, exact_pos, items),
        "two_tower_hnsw": positions_to_recs(users, ann_pos, items),
    }
    for model in (
        PopularityRecommender(weeks=1),
        AgePopularityRecommender(data.customers, weeks=4),
        ItemCooccurrenceRecommender(),
        RepurchaseRecommender(),
    ):
        candidates[model.name] = model.fit(fold.history).recommend(users, k_max)
    tables = []
    for name, recs in candidates.items():
        t = evaluate(recs, truth, CANDIDATE_KS, item_counts, n_boot=ctx.n_boot, seed=ctx.seed)
        t.insert(0, "model", name)
        tables.append(t)
    result = pd.concat(tables, ignore_index=True)

    exact_recall = result.query("model == 'two_tower_exact' and metric == 'recall' and k == 100")
    ann_recall = result.query("model == 'two_tower_hnsw' and metric == 'recall' and k == 100")
    ann = {
        "index_items": len(items),
        "dim": int(item_vecs.shape[1]),
        "hnsw": {"M": 32, "efConstruction": 200, "efSearch": 128},
        "overlap_with_exact@100": faiss_index.overlap_at_k(exact_pos[:, :100], ann_pos[:, :100]),
        "overlap_with_exact@200": faiss_index.overlap_at_k(exact_pos, ann_pos),
        "candidate_recall@100_loss": float(exact_recall.value.iloc[0] - ann_recall.value.iloc[0]),
        "latency_ms_flat": faiss_index.latency_ms(flat, user_vecs, 100),
        "latency_ms_hnsw": faiss_index.latency_ms(hnsw, user_vecs, 100),
    }
    log.info(
        "HNSW overlap@100 %.3f, p95 %.3f ms (exact %.3f ms)",
        ann["overlap_with_exact@100"],
        ann["latency_ms_hnsw"]["p95"],
        ann["latency_ms_flat"]["p95"],
    )
    ctx.write_result(
        "retrieval_validation",
        {
            "split": "validation",
            "eval_users": len(users),
            "cold_start": cold_start_report(fold.history, fold.target),
            "config": cfg.to_dict(),
            "training": {
                "examples_weeks": cfg.train_weeks,
                "best_epoch": retriever.best_epoch_,
                "loss_per_epoch": retriever.losses_,
                "val_recall@100_per_epoch": retriever.val_recall_,
            },
            "ann": ann,
            "metrics": records(result),
        },
    )
    return result

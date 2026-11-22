"""Export everything the API needs into one folder (``artifacts/serving/``).

The bundle is the test-fold state: models and features as of the start of the
test week (history = train + validation), i.e. exactly what was evaluated.
"""

from __future__ import annotations

import json
import logging
import pickle
import shutil
from pathlib import Path

import numpy as np

from recsys.coldstart.users import ColdUserRecommender
from recsys.context import Context
from recsys.data.schema import write_json
from recsys.ranking.candidates import CandidateGenerator
from recsys.retrieval import index as faiss_index
from recsys.retrieval.retriever import TwoTowerRetriever

log = logging.getLogger(__name__)

HISTORY_PER_USER = 20


def export_bundle(ctx: Context) -> Path:
    src = ctx.artifacts_dir / "folds" / "test"
    if not (src / "candidates.pkl").exists():
        raise RuntimeError("run the ranking stage first (it saves the test-fold state)")
    out = ctx.artifacts_dir / "serving"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    data, fold = ctx.data, ctx.folds["test"]

    shutil.copytree(src / "retriever", out / "retriever")
    shutil.copytree(src / "features", out / "features")
    shutil.copy(src / "candidates.pkl", out / "candidates.pkl")
    shutil.copy(ctx.artifacts_dir / "ranking" / "ranker.txt", out / "ranker.txt")

    retriever = TwoTowerRetriever.load(
        src / "retriever", data.articles, data.customers, fold.history
    )
    users = np.sort(fold.history["customer_id"].unique())
    np.save(out / "user_ids.npy", users)
    np.save(out / "user_vectors.npy", retriever.user_vectors(users.tolist(), fold.history))
    gen = CandidateGenerator.load(src / "candidates.pkl")
    faiss_index.save(faiss_index.build_hnsw(gen.item_vecs_), out / "items_hnsw.faiss")
    np.save(out / "index_items.npy", gen.items_)

    cold = {"chosen_alpha": 1.0, "cold_path_validation": {"choice": "cold_blend"}}
    cold_path = ctx.results_dir / "coldstart.json"
    if cold_path.exists():
        cold = json.loads(cold_path.read_text())["cold_users"]
    blend = ColdUserRecommender(data.customers, alpha=cold["chosen_alpha"]).fit(fold.history)
    blend.customers = None
    with (out / "cold_users.pkl").open("wb") as fh:
        pickle.dump(blend, fh, protocol=pickle.HIGHEST_PROTOCOL)

    data.articles.to_parquet(out / "articles.parquet", index=False)  # catalogue + attributes
    data.customers[["customer_id", "age"]].to_parquet(out / "customers.parquet", index=False)
    recent = (
        fold.history.sort_values(["customer_id", "t_dat"])
        .groupby("customer_id")
        .tail(HISTORY_PER_USER)[["customer_id", "t_dat", "article_id", "price"]]
    )
    recent.to_parquet(out / "history.parquet", index=False)
    manifest = {
        "data": ctx.source,
        "as_of": fold.start.date().isoformat(),
        "users_with_embeddings": len(users),
        "index_items": len(gen.items_),
        "catalogue_articles": len(data.articles),
        "cold_users": {
            "alpha": cold["chosen_alpha"],
            "path": cold["cold_path_validation"]["choice"],
        },
        "candidates": vars(gen.config),
    }
    write_json(out / "manifest.json", manifest)
    log.info("serving bundle written to %s", out)
    return out

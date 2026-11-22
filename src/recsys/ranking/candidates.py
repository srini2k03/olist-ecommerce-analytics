"""Candidate generation: the union of four sources, each fit on a fold's history only.

The generator keeps only numpy/pandas state (item vectors, co-occurrence
neighbours, histories, popularity lists), so the API can unpickle it and feed
it user vectors and HNSW results instead of the PyTorch model.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Self

import numpy as np
import pandas as pd

from recsys.baselines import (
    AgePopularityRecommender,
    ItemCooccurrenceRecommender,
    RepurchaseRecommender,
)
from recsys.data.split import check_strictly_before

if TYPE_CHECKING:  # PyTorch stays optional for code that only unpickles a generator
    from recsys.retrieval.retriever import TwoTowerRetriever

log = logging.getLogger(__name__)

SOURCES = ("two_tower", "cooc", "repurchase", "age_pop")
SOURCE_COLUMNS = [f"src_{s}" for s in SOURCES]


@dataclass
class CandidateConfig:
    n_two_tower: int = 100
    n_cooc: int = 30
    n_repurchase: int = 30
    n_age_pop: int = 30


class CandidateGenerator:
    """Two-tower top-N, co-occurrence neighbours, the user's own recent articles and
    age-bucket popularity, with the source(s) of each candidate recorded."""

    def __init__(self, customers: pd.DataFrame, config: CandidateConfig | None = None) -> None:
        self.customers = customers
        self.config = config or CandidateConfig()

    def fit(self, history: pd.DataFrame, as_of: pd.Timestamp, retriever: TwoTowerRetriever) -> Self:
        """``retriever`` must have been trained on ``history`` (or earlier data)."""
        check_strictly_before(history, as_of, "candidate history")
        self.as_of_ = pd.Timestamp(as_of)
        self.items_ = retriever.index_items()
        self.item_vecs_ = retriever.item_vectors(self.items_)
        self._item_pos = pd.Index(self.items_)
        self.cooc_ = ItemCooccurrenceRecommender().fit(history)
        self.repurchase_ = RepurchaseRecommender().fit(history)
        self.age_pop_ = AgePopularityRecommender(self.customers, weeks=4).fit(history)
        # The fitted user->bucket map is kept; drop the frames so the pickle stays small.
        self.age_pop_.customers = None
        self.customers = None
        return self

    def generate_offline(
        self, users: list[int], retriever: TwoTowerRetriever, history: pd.DataFrame
    ) -> pd.DataFrame:
        """Exact two-tower search with fresh user vectors (evaluation)."""
        from recsys.retrieval.retriever import exact_topk

        user_vecs = retriever.user_vectors(users, history)
        ids, _ = exact_topk(user_vecs, self.item_vecs_, self.items_, self.config.n_two_tower)
        return self.generate(users, user_vecs, ids)

    def generate(self, users: list[int], user_vecs: np.ndarray, tt_ids: np.ndarray) -> pd.DataFrame:
        """Union of the four sources for ``users`` given their vectors and two-tower top-N."""
        cfg = self.config
        users = [int(u) for u in users]
        frames = []
        if tt_ids.size:
            frames.append(
                pd.DataFrame(
                    {
                        "customer_id": np.repeat(users, tt_ids.shape[1]),
                        "article_id": tt_ids.ravel(),
                        "tt_rank": np.tile(np.arange(1, tt_ids.shape[1] + 1), len(users)),
                        "src_two_tower": 1,
                    }
                )
            )
        rows = []
        for u in users:
            scores = self.cooc_.neighbour_scores(u)
            top = sorted(scores, key=lambda b: (-scores[b], b))[: cfg.n_cooc]
            rows += [(u, a, scores[a]) for a in top]
        cooc = pd.DataFrame(rows, columns=["customer_id", "article_id", "cooc_score"])
        frames.append(cooc.assign(src_cooc=1))
        rows = [
            (u, a)
            for u in users
            for a, _, _ in self.repurchase_.histories_.get(u, [])[: cfg.n_repurchase]
        ]
        frames.append(
            pd.DataFrame(rows, columns=["customer_id", "article_id"]).assign(src_repurchase=1)
        )
        pop = self.age_pop_.recommend(users, cfg.n_age_pop)
        rows = [(u, a, r + 1) for u, items in pop.items() for r, a in enumerate(items)]
        frames.append(
            pd.DataFrame(rows, columns=["customer_id", "article_id", "agepop_rank"]).assign(
                src_age_pop=1
            )
        )
        cands = (
            pd.concat(frames, ignore_index=True)
            .groupby(["customer_id", "article_id"], sort=True)
            .agg(
                tt_rank=("tt_rank", "min"),
                cooc_score=("cooc_score", "max"),
                agepop_rank=("agepop_rank", "min"),
                **{c: (c, "max") for c in SOURCE_COLUMNS},
            )
            .reset_index()
        )
        # Empty per-source frames (e.g. one user without co-occurrences) arrive as object
        # columns; pin every dtype so features are numeric whatever the batch.
        cands = cands.astype({"customer_id": np.int64, "article_id": np.int64})
        num = ["tt_rank", "cooc_score", "agepop_rank"]
        cands[num] = cands[num].astype(np.float64)
        cands[SOURCE_COLUMNS] = cands[SOURCE_COLUMNS].fillna(0).astype(np.int8)
        cands["tt_score"] = self._tt_scores(cands, users, user_vecs)
        cands["sources"] = [
            "|".join(s for s, flag in zip(SOURCES, row, strict=True) if flag)
            for row in cands[SOURCE_COLUMNS].to_numpy()
        ]
        return cands

    def _tt_scores(
        self, cands: pd.DataFrame, users: list[int], user_vecs: np.ndarray
    ) -> np.ndarray:
        """Two-tower cosine score for every candidate, whichever source produced it."""
        out = np.full(len(cands), np.nan, dtype=np.float32)
        ui = pd.Index(users).get_indexer(cands["customer_id"])
        ii = self._item_pos.get_indexer(cands["article_id"])
        ok = np.flatnonzero((ui >= 0) & (ii >= 0))
        step = 500_000  # bounded memory on the real data (millions of candidates)
        for s in range(0, len(ok), step):
            rows = ok[s : s + step]
            out[rows] = np.einsum("ij,ij->i", user_vecs[ui[rows]], self.item_vecs_[ii[rows]])
        return out

    def item_vectors(self, article_ids: np.ndarray) -> np.ndarray:
        return self.item_vecs_[self._item_pos.get_indexer(article_ids)]

    # ------------------------------------------------------------------ persistence

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: Path) -> CandidateGenerator:
        """Only for artifacts this pipeline wrote itself (pickle runs code on load)."""
        with path.open("rb") as fh:
            return pickle.load(fh)  # noqa: S301 (trusted, self-produced artifact)

"""In-process recommender: loads a serving bundle once and answers requests.

Known customers: precomputed user vector -> HNSW top-N -> candidate union ->
features -> LambdaRank. Customers with history but no stored vector get one
from the user tower on the fly (needs PyTorch). Customers with no history
(including unknown ids) take the cold-start path chosen on validation: the
age-popularity/trending blend, or the ranker fed by the user tower's
attribute-only (OOV id + age) vector.
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.data.schema import hash_customer_ids
from recsys.ranking.candidates import CandidateGenerator
from recsys.ranking.features import FeatureStore, assemble
from recsys.ranking.ranker import LambdaRanker
from recsys.retrieval import index as faiss_index

log = logging.getLogger(__name__)

ATTRIBUTES = [
    "prod_name",
    "product_type_name",
    "product_group_name",
    "colour_group_name",
    "department_name",
    "index_group_name",
    "garment_group_name",
]


def parse_customer_id(raw: str | int) -> int:
    """Accept the processed int64 id or the raw 64-character hex id from the CSVs."""
    if isinstance(raw, int):
        return raw
    s = str(raw).strip()
    if len(s) == 64 and all(c in "0123456789abcdefABCDEF" for c in s):
        return int(hash_customer_ids([s.lower()])[0])
    return int(s)


class RecommenderService:
    def __init__(self, folder: Path, load_torch: bool = True) -> None:
        t0 = time.perf_counter()
        self.folder = folder
        self.manifest = json.loads((folder / "manifest.json").read_text())
        self.user_ids = np.load(folder / "user_ids.npy")
        self.user_vecs = np.load(folder / "user_vectors.npy")
        self._user_row = pd.Index(self.user_ids)
        self.items = np.load(folder / "index_items.npy")
        self.index = faiss_index.load(folder / "items_hnsw.faiss")
        self.generator = CandidateGenerator.load(folder / "candidates.pkl")
        self.store = FeatureStore.load(folder / "features")
        self.ranker = LambdaRanker.load(folder / "ranker.txt")
        with (folder / "cold_users.pkl").open("rb") as fh:
            self.cold = pickle.load(fh)  # noqa: S301 (trusted, self-produced artifact)
        self.articles_full = pd.read_parquet(folder / "articles.parquet")
        self.articles = self.articles_full.set_index("article_id")
        customers = pd.read_parquet(folder / "customers.parquet")
        self.ages = customers.set_index("customer_id")["age"]
        self.history = pd.read_parquet(folder / "history.parquet")
        self._history_rows = self.history.groupby("customer_id").indices
        self.cold_path = self.manifest["cold_users"]["path"]
        self.retriever = self._load_retriever(customers) if load_torch else None
        log.info("loaded serving bundle from %s in %.1f s", folder, time.perf_counter() - t0)

    def _load_retriever(self, customers: pd.DataFrame):
        try:
            from recsys.retrieval.retriever import TwoTowerRetriever
        except ImportError:  # PyTorch is optional: cold users then use the blend
            log.warning("PyTorch not installed: no on-the-fly user vectors")
            return None
        return TwoTowerRetriever.load(
            self.folder / "retriever", self.articles_full, customers, self.history
        )

    # ------------------------------------------------------------------ requests

    def user_vector(self, cid: int, age: float | None) -> tuple[np.ndarray | None, str]:
        row = self._user_row.get_indexer([cid])[0]
        if row >= 0:
            return self.user_vecs[row], "two_tower+ranker"
        if self.retriever is None:
            return None, "cold_blend"
        if cid in self._history_rows:
            hist = self.history.iloc[self._history_rows[cid]]
            return self.retriever.user_vectors([cid], hist)[0], "two_tower(on the fly)+ranker"
        if self.cold_path == "ranker":
            empty = self.history.iloc[:0]
            return self.retriever.user_vectors([cid], empty, ages=[age])[0], "cold:two_tower+ranker"
        return None, "cold_blend"

    def recommend(self, customer: str | int, k: int = 12, age: float | None = None) -> dict:
        t0 = time.perf_counter()
        cid = parse_customer_id(customer)
        if age is None:
            known = self.ages.get(cid)
            age = None if known is None or pd.isna(known) else float(known)
        vec, strategy = self.user_vector(cid, age)
        fallback = self.cold.recommend_for_age(age, max(k, 50))
        if vec is None:
            items = [
                {"article_id": int(a), "score": float(1 / (r + 1)), "sources": "cold_blend"}
                for r, a in enumerate(fallback[:k])
            ]
        else:
            items = self._ranked(cid, vec, k)
            seen = {i["article_id"] for i in items}
            for a in fallback:
                if len(items) >= k:
                    break
                if a not in seen:
                    items.append(
                        {"article_id": int(a), "score": None, "sources": "popularity_fill"}
                    )
        for i in items:
            i.update(self.attributes(i["article_id"]) or {})
        return {
            "customer_id": str(cid),
            "strategy": strategy,
            "k": k,
            "items": items,
            "latency_ms": (time.perf_counter() - t0) * 1000,
        }

    def _ranked(self, cid: int, vec: np.ndarray, k: int) -> list[dict]:
        n = self.generator.config.n_two_tower
        _, pos = faiss_index.search(self.index, vec[None, :], n)
        ids = self.items[pos[0][pos[0] >= 0]][None, :]
        cands = self.generator.generate([cid], vec[None, :], ids)
        feats = assemble(cands, self.store)
        feats["score"] = self.ranker.predict(feats)
        top = feats.sort_values(["score", "article_id"], ascending=[False, True]).head(k)
        return [
            {"article_id": int(a), "score": float(s), "sources": src}
            for a, s, src in zip(top["article_id"], top["score"], top["sources"], strict=True)
        ]

    def attributes(self, article_id: int) -> dict | None:
        if article_id not in self.articles.index:
            return None
        row = self.articles.loc[article_id]
        return {c: (None if pd.isna(row[c]) else str(row[c])) for c in ATTRIBUTES if c in row}

    def item(self, article_id: int) -> dict | None:
        attrs = self.attributes(article_id)
        if attrs is None:
            return None
        pop = self.store.items["i_pop_1w"].get(article_id, 0.0)
        return {
            "article_id": article_id,
            "product_code": int(self.articles.loc[article_id, "product_code"]),
            "purchases_last_week": float(pop),
            **attrs,
        }

    def user_history(self, customer: str | int) -> list[dict]:
        cid = parse_customer_id(customer)
        rows = self._history_rows.get(cid)
        if rows is None:
            return []
        hist = self.history.iloc[rows].sort_values("t_dat", ascending=False)
        return [
            {
                "t_dat": t.date().isoformat(),
                "article_id": int(a),
                "price": float(p),
                **(self.attributes(int(a)) or {}),
            }
            for t, a, p in zip(hist["t_dat"], hist["article_id"], hist["price"], strict=True)
        ]

    def sample_users(self, n: int = 10, seed: int = 0) -> list[str]:
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(self.user_ids), min(n, len(self.user_ids)), replace=False)
        return [str(u) for u in self.user_ids[np.sort(pick)]]

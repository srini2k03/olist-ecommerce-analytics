"""Training, inference and persistence for the two-tower retriever."""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
import torch

from recsys.baselines.base import Recommender
from recsys.baselines.popularity import AGE_BINS
from recsys.data.folds import Fold
from recsys.evaluate.metrics import recall_at_k
from recsys.retrieval.model import TwoTowerConfig, TwoTowerModel, in_batch_loss
from recsys.retrieval.vocab import ITEM_ATTRS, Catalogue, Vocab, age_bucket_index, history_matrix

log = logging.getLogger(__name__)

DAY = pd.Timedelta(days=1)
WEEK = pd.Timedelta(days=7)
CHUNK = 2048
N_AGES = len(AGE_BINS)  # len(bins) - 1 buckets + index 0 for unknown age


class TwoTowerRetriever(Recommender):
    """``fit(train)`` learns the towers from ``train`` only; ``topk`` / ``recommend``
    score users against every article seen in training (optionally also unseen
    catalogue articles, embedded through the attribute-only path)."""

    name = "two_tower"

    def __init__(
        self,
        articles: pd.DataFrame,
        customers: pd.DataFrame,
        config: TwoTowerConfig | None = None,
    ) -> None:
        self.articles = articles
        self.customers = customers
        self.config = config or TwoTowerConfig()

    # ------------------------------------------------------------------ data

    def _ages(self, users: Sequence[int]) -> np.ndarray:
        age = self.customers.set_index("customer_id")["age"]
        return age_bucket_index(age.reindex(pd.Index(users)))

    def _examples(self, train: pd.DataFrame) -> dict[str, np.ndarray]:
        """(user state before week w, item bought in week w) for the last ``train_weeks`` weeks."""
        cfg = self.config
        end = train["t_dat"].max() + DAY
        parts = []
        for w in range(cfg.train_weeks):
            start = end - WEEK * (w + 1)
            week = train[(train["t_dat"] >= start) & (train["t_dat"] < start + WEEK)]
            if week.empty:
                continue
            before = train[train["t_dat"] < start]
            pairs = week[["customer_id", "article_id"]].drop_duplicates()
            users = pairs["customer_id"].unique()
            hist = history_matrix(before, users, cfg.history_len, self.item_vocab_)
            row = pd.Index(users).get_indexer(pairs["customer_id"])
            parts.append(
                {
                    "user": self.user_vocab_.encode(pairs["customer_id"]),
                    "age": self._ages(users)[row],
                    "hist": hist[row],
                    "item": self.item_vocab_.encode(pairs["article_id"]),
                }
            )
        if not parts:
            raise ValueError("no training weeks in the history")
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}

    # ------------------------------------------------------------------ training

    def fit(
        self, train: pd.DataFrame, val_fold: Fold | None = None, epochs: int | None = None
    ) -> Self:
        cfg = self.config
        torch.manual_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        self.history_ = train
        self.item_vocab_ = Vocab(train["article_id"])
        self.user_vocab_ = Vocab(train["customer_id"])
        self.catalogue_ = Catalogue(self.articles)
        ex = self._examples(train)
        n_items = len(self.item_vocab_)
        counts = np.bincount(ex["item"], minlength=n_items).astype(np.float64)
        self.log_q_ = torch.tensor(
            np.log((counts + 1.0) / (counts.sum() + n_items)), dtype=torch.float32
        )
        self.model_ = TwoTowerModel(
            len(self.user_vocab_), n_items, N_AGES, self.catalogue_.attr_sizes(), cfg
        )
        self._item_attrs = torch.from_numpy(self._attr_table())
        opt = torch.optim.AdamW(self.model_.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        tensors = {k: torch.from_numpy(v) for k, v in ex.items()}
        n = len(ex["item"])
        max_epochs = epochs or cfg.epochs
        self.losses_, self.val_recall_ = [], []
        best, best_state, stale = -1.0, None, 0
        for epoch in range(max_epochs):
            self.model_.train()
            perm = torch.from_numpy(rng.permutation(n))
            total, batches = 0.0, 0
            for s in range(0, n, cfg.batch_size):
                b = perm[s : s + cfg.batch_size]
                if len(b) < 2:
                    continue
                user, item = tensors["user"][b], tensors["item"][b]
                user_in = torch.where(torch.rand(len(b)) < cfg.user_id_dropout, 0, user)
                item_in = torch.where(torch.rand(len(b)) < cfg.item_id_dropout, 0, item)
                u = self.model_.user_vec(user_in, tensors["age"][b], tensors["hist"][b])
                v = self.model_.item_vec(item_in, self._item_attrs[item])
                log_q = self.log_q_[item] if cfg.logq else None
                loss = in_batch_loss(u, v, item, log_q, cfg.temperature)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item()
                batches += 1
            self.losses_.append(total / max(batches, 1))
            self.fitted_ = True
            msg = f"epoch {epoch + 1}: loss {self.losses_[-1]:.4f}"
            if val_fold is not None:
                recall = self.evaluate_recall(val_fold, cfg.eval_k)
                self.val_recall_.append(recall)
                msg += f", validation recall@{cfg.eval_k} {recall:.4f}"
                if recall > best:
                    best, best_state, stale = recall, copy.deepcopy(self.model_.state_dict()), 0
                    self.best_epoch_ = epoch + 1
                else:
                    stale += 1
            log.info(msg)
            if val_fold is not None and stale >= cfg.patience:
                break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        else:
            self.best_epoch_ = len(self.losses_)
        self.model_.eval()
        return self

    def _attr_table(self) -> np.ndarray:
        """Attributes per item-vocab index (row 0, the OOV item, is all zeros)."""
        attrs = self.catalogue_.encode(self.item_vocab_.values.to_numpy())
        return np.vstack([np.zeros((1, len(ITEM_ATTRS)), dtype=np.int64), attrs])

    # ------------------------------------------------------------------ inference

    @torch.no_grad()
    def user_vectors(
        self,
        user_ids: Sequence[int],
        history: pd.DataFrame | None = None,
        ages: Sequence[float | None] | None = None,
    ) -> np.ndarray:
        """``ages`` overrides the customers table (e.g. an age given to the API)."""
        self._check_fitted()
        history = self.history_ if history is None else history
        users = list(user_ids)
        hist = history_matrix(history, users, self.config.history_len, self.item_vocab_)
        if ages is None:
            age_idx = self._ages(users)
        else:
            age_idx = age_bucket_index(pd.Series(ages, dtype="float64"))
        out = self.model_.user_vec(
            torch.from_numpy(self.user_vocab_.encode(users)),
            torch.from_numpy(age_idx),
            torch.from_numpy(hist),
        )
        return out.numpy().astype(np.float32)

    @torch.no_grad()
    def item_vectors(self, article_ids: Sequence[int]) -> np.ndarray:
        """Unseen articles use the OOV id embedding, i.e. the attribute-only path."""
        self._check_fitted()
        ids = np.asarray(article_ids)
        out = self.model_.item_vec(
            torch.from_numpy(self.item_vocab_.encode(ids)),
            torch.from_numpy(self.catalogue_.encode(ids)),
        )
        return out.numpy().astype(np.float32)

    def index_items(self, include_unseen: bool = False) -> np.ndarray:
        """Articles to retrieve from: those seen in training, plus (optionally) the rest
        of the catalogue."""
        seen = self.item_vocab_.values.to_numpy()
        if not include_unseen:
            return seen
        unseen = self.catalogue_.article_ids.difference(pd.Index(seen)).to_numpy()
        return np.concatenate([seen, unseen])

    def topk(
        self,
        user_ids: Sequence[int],
        k: int,
        history: pd.DataFrame | None = None,
        items: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Exact top-``k`` (article ids, cosine scores) per user, best first."""
        items = self.index_items() if items is None else np.asarray(items)
        return exact_topk(self.user_vectors(user_ids, history), self.item_vectors(items), items, k)

    def recommend(
        self, user_ids: Iterable[int], k: int, history: pd.DataFrame | None = None
    ) -> dict[int, list[int]]:
        users = [int(u) for u in user_ids]
        ids, _ = self.topk(users, k, history)
        return {u: [int(i) for i in row] for u, row in zip(users, ids, strict=True)}

    def evaluate_recall(self, fold: Fold, k: int) -> float:
        truth = fold.truth
        recs = self.recommend(sorted(truth), k, fold.history)
        return float(np.mean([recall_at_k(recs[u], rel, k) for u, rel in truth.items()]))

    # ------------------------------------------------------------------ persistence

    def save(self, folder: Path) -> None:
        self._check_fitted()
        folder.mkdir(parents=True, exist_ok=True)
        torch.save(self.model_.state_dict(), folder / "model.pt")
        np.savez(
            folder / "vocab.npz",
            users=self.user_vocab_.values.to_numpy(np.int64),
            items=self.item_vocab_.values.to_numpy(np.int64),
        )
        meta = {
            "config": self.config.to_dict(),
            "best_epoch": self.best_epoch_,
            "losses": self.losses_,
            "val_recall": self.val_recall_,
            "attr_vocabs": {a: self.catalogue_.vocabs[a].to_list() for a in ITEM_ATTRS},
        }
        (folder / "retriever.json").write_text(json.dumps(meta, indent=1))

    @classmethod
    def load(
        cls,
        folder: Path,
        articles: pd.DataFrame,
        customers: pd.DataFrame,
        history: pd.DataFrame | None = None,
    ) -> TwoTowerRetriever:
        meta = json.loads((folder / "retriever.json").read_text())
        obj = cls(articles, customers, TwoTowerConfig(**meta["config"]))
        vocab = np.load(folder / "vocab.npz")
        obj.user_vocab_, obj.item_vocab_ = Vocab(vocab["users"]), Vocab(vocab["items"])
        obj.catalogue_ = Catalogue(articles, {a: Vocab(v) for a, v in meta["attr_vocabs"].items()})
        obj.model_ = TwoTowerModel(
            len(obj.user_vocab_),
            len(obj.item_vocab_),
            N_AGES,
            obj.catalogue_.attr_sizes(),
            obj.config,
        )
        obj.model_.load_state_dict(torch.load(folder / "model.pt", weights_only=True))
        obj.model_.eval()
        obj.best_epoch_, obj.losses_, obj.val_recall_ = (
            meta["best_epoch"],
            meta["losses"],
            meta["val_recall"],
        )
        obj.history_ = (
            history
            if history is not None
            else pd.DataFrame(columns=["t_dat", "customer_id", "article_id"])
        )
        obj.fitted_ = True
        return obj


def exact_topk(
    user_vecs: np.ndarray, item_vecs: np.ndarray, items: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Brute-force top-``k`` by inner product, in user chunks to bound memory."""
    v = torch.from_numpy(np.ascontiguousarray(item_vecs, dtype=np.float32))
    u = torch.from_numpy(np.ascontiguousarray(user_vecs, dtype=np.float32))
    k = min(k, len(items))
    ids, scores = [], []
    for s in range(0, len(u), CHUNK):
        sc, ix = torch.topk(u[s : s + CHUNK] @ v.T, k, dim=1)
        ids.append(items[ix.numpy()])
        scores.append(sc.numpy())
    if not ids:
        return np.empty((0, k), dtype=items.dtype), np.empty((0, k), dtype=np.float32)
    return np.vstack(ids), np.vstack(scores)

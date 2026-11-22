"""Item-item co-occurrence: "customers who bought this also bought" within a time window."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Self

import pandas as pd

from recsys.baselines.base import Recommender, fill, recent_window, user_histories
from recsys.baselines.popularity import PopularityRecommender


class ItemCooccurrenceRecommender(Recommender):
    """Neighbours of the user's recent articles, topped up with popularity.

    Two articles co-occur when one customer bought both within ``window_days``
    of each other; the pair's weight is the number of such customers. A user's
    candidates are the neighbours of their ``n_seed_items`` most recent articles,
    weighted by ``1 / (1 + recency rank of the seed)``. The seed articles
    themselves are excluded (that is the repurchase baseline's job).
    """

    def __init__(
        self,
        window_days: int = 7,
        pair_weeks: int | None = 12,
        n_seed_items: int = 10,
        n_neighbours: int = 50,
        max_items_per_user: int = 30,
        popularity_weeks: int | None = 1,
    ) -> None:
        self.window_days = window_days
        self.pair_weeks = pair_weeks
        self.n_seed_items = n_seed_items
        self.n_neighbours = n_neighbours
        self.max_items_per_user = max_items_per_user
        self.popularity_weeks = popularity_weeks
        self.name = "item_cooccurrence"

    def _pairs(self, train: pd.DataFrame) -> pd.DataFrame:
        events = (
            recent_window(train, self.pair_weeks)[["customer_id", "article_id", "t_dat"]]
            .drop_duplicates()
            .sort_values(["customer_id", "t_dat", "article_id"], ascending=[True, False, True])
        )
        # Bound the self-join: only each customer's most recent purchases.
        events = events.groupby("customer_id").head(self.max_items_per_user)
        joined = events.merge(events, on="customer_id", suffixes=("_a", "_b"))
        close = (joined["t_dat_a"] - joined["t_dat_b"]).abs() <= pd.Timedelta(days=self.window_days)
        joined = joined[close & (joined["article_id_a"] != joined["article_id_b"])]
        pairs = (
            joined[["customer_id", "article_id_a", "article_id_b"]]
            .drop_duplicates()
            .groupby(["article_id_a", "article_id_b"])
            .size()
            .rename("weight")
            .reset_index()
            .sort_values(["article_id_a", "weight", "article_id_b"], ascending=[True, False, True])
        )
        return pairs.groupby("article_id_a").head(self.n_neighbours)

    def fit(self, train: pd.DataFrame) -> Self:
        pairs = self._pairs(train)
        self.neighbours_: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for a, b, w in pairs.itertuples(index=False, name=None):
            self.neighbours_[int(a)].append((int(b), int(w)))
        self.histories_ = user_histories(train)
        self.popular_ = PopularityRecommender(self.popularity_weeks).fit(train).ranking_
        self.fitted_ = True
        return self

    def seeds(self, user: int) -> list[int]:
        return [a for a, _, _ in self.histories_.get(int(user), [])[: self.n_seed_items]]

    def neighbour_scores(self, user: int) -> dict[int, float]:
        """Recency-weighted co-occurrence score of every neighbour of the user's seeds."""
        self._check_fitted()
        seeds = self.seeds(user)
        scores: dict[int, float] = defaultdict(float)
        for rank, seed in enumerate(seeds):
            for b, w in self.neighbours_.get(seed, ()):
                scores[b] += w / (1 + rank)
        for seed in seeds:
            scores.pop(seed, None)
        return dict(scores)

    def recommend(self, user_ids: Iterable[int], k: int) -> dict[int, list[int]]:
        self._check_fitted()
        out = {}
        for u in user_ids:
            scores = self.neighbour_scores(u)
            ranked = sorted(scores, key=lambda b: (-scores[b], b))
            out[int(u)] = fill(ranked, self.popular_, k, exclude=self.seeds(u))
        return out

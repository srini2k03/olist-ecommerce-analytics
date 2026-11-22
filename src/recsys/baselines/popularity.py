"""Popularity baselines: global, and per age bucket."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Self

import pandas as pd

from recsys.baselines.base import Recommender, fill, popularity_ranking, recent_window

AGE_BINS = (0, 20, 25, 30, 40, 50, 60, 200)


class PopularityRecommender(Recommender):
    """The most purchased articles in the last ``weeks`` weeks of training data."""

    def __init__(self, weeks: int | None = 1) -> None:
        self.weeks = weeks
        self.name = f"popularity_{weeks}w" if weeks else "popularity_all"

    def fit(self, train: pd.DataFrame) -> Self:
        recent = popularity_ranking(recent_window(train, self.weeks))
        # Top up from all-time popularity so k items are always available.
        self.ranking_ = fill(recent, popularity_ranking(train), len(set(train["article_id"])))
        self.fitted_ = True
        return self

    def recommend(self, user_ids: Iterable[int], k: int) -> dict[int, list[int]]:
        self._check_fitted()
        top = self.ranking_[:k]
        return {int(u): list(top) for u in user_ids}


class AgePopularityRecommender(Recommender):
    """Popularity within the user's age bucket; unknown age falls back to global popularity.

    Ages come from the customers table given at construction (not from the
    transactions), so the model needs no information from the target window.
    """

    def __init__(
        self,
        customers: pd.DataFrame,
        weeks: int | None = 4,
        bins: Sequence[int] = AGE_BINS,
        min_bucket_rows: int = 50,
    ) -> None:
        self.customers = customers
        self.weeks = weeks
        self.bins = tuple(bins)
        self.min_bucket_rows = min_bucket_rows
        self.name = f"age_popularity_{weeks}w" if weeks else "age_popularity_all"

    def _buckets(self, ages: pd.Series) -> pd.Series:
        return pd.cut(ages, bins=list(self.bins), right=False, labels=False)

    def fit(self, train: pd.DataFrame) -> Self:
        bucket = self._buckets(self.customers.set_index("customer_id")["age"])
        self.user_bucket_ = {int(u): int(b) for u, b in bucket.dropna().items()}
        self.global_ = PopularityRecommender(self.weeks).fit(train).ranking_
        recent = recent_window(train, self.weeks)
        recent_bucket = recent["customer_id"].map(self.user_bucket_)
        self.bucket_rankings_: dict[int, list[int]] = {}
        for b, rows in recent.groupby(recent_bucket):
            if len(rows) >= self.min_bucket_rows:
                self.bucket_rankings_[int(b)] = popularity_ranking(rows)
        self.fitted_ = True
        return self

    def recommend(self, user_ids: Iterable[int], k: int) -> dict[int, list[int]]:
        self._check_fitted()
        cache: dict[int | None, list[int]] = {None: self.global_[:k]}
        out = {}
        for u in user_ids:
            b = self.user_bucket_.get(int(u))
            if b not in self.bucket_rankings_:
                b = None
            if b not in cache:
                cache[b] = fill(self.bucket_rankings_[b], self.global_, k)
            out[int(u)] = list(cache[b])
        return out

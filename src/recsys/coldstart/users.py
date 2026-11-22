"""New users: a blend of age-bucket popularity and last-week trending articles."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Self

import numpy as np
import pandas as pd

from recsys.baselines.base import WEEK, Recommender, recent_window
from recsys.baselines.popularity import AGE_BINS


def _normalise(counts: pd.Series) -> pd.Series:
    return counts / counts.max() if len(counts) and counts.max() > 0 else counts


class ColdUserRecommender(Recommender):
    """``score = alpha * age_popularity + (1 - alpha) * trending``, both scaled to [0, 1].

    * age popularity: purchases in the user's age bucket over the last ``age_weeks``
      (global popularity when the age is unknown or the bucket is thin);
    * trending: last-week purchases times last-week growth,
      ``n_1w * (n_1w + 1) / (n_prev_week + 1)``, which favours articles that are
      both selling and accelerating.
    """

    def __init__(
        self,
        customers: pd.DataFrame,
        alpha: float = 0.5,
        age_weeks: int = 4,
        bins: Sequence[int] = AGE_BINS,
        min_bucket_rows: int = 50,
        pool: int = 500,
    ) -> None:
        self.customers = customers
        self.alpha = alpha
        self.age_weeks = age_weeks
        self.bins = tuple(bins)
        self.min_bucket_rows = min_bucket_rows
        self.pool = pool
        self.name = f"cold_blend_a{alpha:g}"

    def bucket(self, age: float | None) -> int | None:
        if age is None or pd.isna(age):
            return None
        b = pd.cut([age], bins=list(self.bins), right=False, labels=False)[0]
        return None if pd.isna(b) else int(b)

    def fit(self, train: pd.DataFrame) -> Self:
        end = train["t_dat"].max() + pd.Timedelta(days=1)
        last = train[train["t_dat"] >= end - WEEK].groupby("article_id").size()
        prev = train[(train["t_dat"] >= end - 2 * WEEK) & (train["t_dat"] < end - WEEK)]
        prev = prev.groupby("article_id").size().reindex(last.index).fillna(0)
        self.trend_ = _normalise(last * (last + 1) / (prev + 1))
        recent = recent_window(train, self.age_weeks)
        ages = self.customers.set_index("customer_id")["age"]
        self.age_of_ = ages.to_dict()
        buckets = pd.cut(
            recent["customer_id"].map(ages), list(self.bins), right=False, labels=False
        )
        self.global_ = _normalise(recent.groupby("article_id").size())
        self.bucket_pop_ = {}
        for b, rows in recent.groupby(buckets):
            if len(rows) >= self.min_bucket_rows:
                self.bucket_pop_[int(b)] = _normalise(rows.groupby("article_id").size())
        self._cache: dict[int | None, list[int]] = {}
        self.fitted_ = True
        return self

    def ranking(self, bucket: int | None) -> list[int]:
        """Blended ranking for an age bucket (``None`` = unknown age)."""
        if bucket not in self._cache:
            pop = self.bucket_pop_.get(bucket, self.global_) if bucket is not None else self.global_
            idx = pop.index.union(self.trend_.index)
            score = self.alpha * pop.reindex(idx).fillna(0) + (
                1 - self.alpha
            ) * self.trend_.reindex(idx).fillna(0)
            order = np.lexsort((idx.to_numpy(), -score.to_numpy()))
            self._cache[bucket] = idx.to_numpy()[order][: self.pool].astype(int).tolist()
        return self._cache[bucket]

    def recommend_for_age(self, age: float | None, k: int) -> list[int]:
        self._check_fitted()
        return self.ranking(self.bucket(age))[:k]

    def recommend(self, user_ids: Iterable[int], k: int) -> dict[int, list[int]]:
        self._check_fitted()
        return {int(u): self.recommend_for_age(self.age_of_.get(int(u)), k) for u in user_ids}

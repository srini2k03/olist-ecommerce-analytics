"""Repurchase: recommend the user's own recent articles, then popular ones."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Self

import pandas as pd

from recsys.baselines.base import Recommender, fill, recent_window, user_histories
from recsys.baselines.popularity import PopularityRecommender


class RepurchaseRecommender(Recommender):
    """The user's articles from the last ``weeks`` weeks, by recency then frequency.

    Ranking key: last purchase date (newest first), then number of purchases,
    then article id. Users with no history get global popularity.
    """

    def __init__(self, weeks: int | None = 8, popularity_weeks: int | None = 1) -> None:
        self.weeks = weeks
        self.popularity_weeks = popularity_weeks
        self.name = "repurchase"

    def fit(self, train: pd.DataFrame) -> Self:
        self.histories_ = user_histories(recent_window(train, self.weeks))
        self.popular_ = PopularityRecommender(self.popularity_weeks).fit(train).ranking_
        self.fitted_ = True
        return self

    def recommend(self, user_ids: Iterable[int], k: int) -> dict[int, list[int]]:
        self._check_fitted()
        out = {}
        for u in user_ids:
            own = [a for a, _, _ in self.histories_.get(int(u), [])]
            out[int(u)] = fill(own, self.popular_, k)
        return out

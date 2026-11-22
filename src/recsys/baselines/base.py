"""Common interface and helpers for the baselines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from itertools import chain
from typing import Self

import pandas as pd

WEEK = pd.Timedelta(days=7)


class Recommender(ABC):
    """``fit(train_df)`` on transactions, then ``recommend(user_ids, k)``.

    ``recommend`` returns exactly ``k`` distinct article ids per user (fewer only
    if the training data has fewer than ``k`` articles), best first.
    """

    name: str = "recommender"

    @abstractmethod
    def fit(self, train: pd.DataFrame) -> Self: ...

    @abstractmethod
    def recommend(self, user_ids: Iterable[int], k: int) -> dict[int, list[int]]: ...

    def _check_fitted(self) -> None:
        if not getattr(self, "fitted_", False):
            raise RuntimeError(f"{type(self).__name__} is not fitted")


def recent_window(train: pd.DataFrame, weeks: int | None) -> pd.DataFrame:
    """Rows in the last ``weeks`` weeks of ``train`` (all rows when ``weeks`` is None)."""
    if weeks is None or train.empty:
        return train
    start = train["t_dat"].max() - WEEK * weeks + pd.Timedelta(days=1)
    return train[train["t_dat"] >= start]


def popularity_ranking(train: pd.DataFrame) -> list[int]:
    """Articles by purchase count, ties broken by article id (deterministic)."""
    counts = train.groupby("article_id").size().rename("n").reset_index()
    counts = counts.sort_values(["n", "article_id"], ascending=[False, True])
    return counts["article_id"].astype(int).tolist()


def fill(
    primary: Sequence[int],
    fallback: Sequence[int],
    k: int,
    exclude: Iterable[int] = (),
) -> list[int]:
    """``primary`` (deduplicated, in order) topped up from ``fallback`` to ``k`` items.

    Items in ``exclude`` are skipped in both lists.
    """
    out: list[int] = []
    seen: set[int] = set(exclude)
    for item in chain(primary, fallback):
        if item not in seen:
            seen.add(item)
            out.append(item)
            if len(out) == k:
                break
    return out


def user_histories(train: pd.DataFrame) -> dict[int, list[tuple[int, pd.Timestamp, int]]]:
    """Per user: (article, last purchase date, purchase count), most recent first.

    Ties on date are broken by count (desc) and then article id.
    """
    agg = (
        train.groupby(["customer_id", "article_id"])["t_dat"]
        .agg(["max", "size"])
        .reset_index()
        .sort_values(
            ["customer_id", "max", "size", "article_id"], ascending=[True, False, False, True]
        )
    )
    out: dict[int, list[tuple[int, pd.Timestamp, int]]] = {}
    for user, art, last, n in agg.itertuples(index=False, name=None):
        out.setdefault(int(user), []).append((int(art), last, int(n)))
    return out

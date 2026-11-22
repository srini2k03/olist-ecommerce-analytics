"""Point-in-time folds: a target week plus all history strictly before it.

Three folds are derived from a :class:`~recsys.data.split.Split`:

* ``ranker_train``: target = the last training week, history = train before it.
  The ranker learns from its labels, with candidates and features built from
  its history only.
* ``validation``: target = the validation week, history = train.
* ``test``: target = the test week, history = train + validation.

Every model that produces candidates or features for a fold is fit on that
fold's history, so no fold ever sees its own target week.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from recsys.data.split import Split, check_strictly_before, check_within, ground_truth

WEEK = pd.Timedelta(days=7)
FOLD_NAMES = ("ranker_train", "validation", "test")


@dataclass
class Fold:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    history: pd.DataFrame
    target: pd.DataFrame

    def __post_init__(self) -> None:
        check_strictly_before(self.history, self.start, f"{self.name} history")
        check_within(self.target, self.start, self.end, f"{self.name} target")

    @property
    def truth(self) -> dict[int, set[int]]:
        return ground_truth(self.target)

    @property
    def users(self) -> list[int]:
        return sorted(self.truth)


def make_folds(split: Split) -> dict[str, Fold]:
    b = split.boundaries
    rt_start = b.val_start - WEEK
    train = split.train
    folds = {
        "ranker_train": Fold(
            "ranker_train",
            rt_start,
            b.val_start,
            train[train["t_dat"] < rt_start].reset_index(drop=True),
            train[train["t_dat"] >= rt_start].reset_index(drop=True),
        ),
        "validation": Fold(
            "validation",
            b.val_start,
            b.test_start,
            split.history_for("validation"),
            split.validation,
        ),
        "test": Fold("test", b.test_start, b.test_end, split.history_for("test"), split.test),
    }
    if folds["ranker_train"].history.empty:
        raise ValueError("train split is too short: no history before the ranker-training week")
    return folds

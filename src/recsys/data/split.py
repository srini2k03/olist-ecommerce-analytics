"""Week-based train / validation / test split and the leakage checks around it.

Weeks are counted back from the last transaction date, as in the competition
(whose hidden test set is the 7 days after the training data ends):

* test       = the final ``test_weeks`` weeks,
* validation = the ``val_weeks`` weeks before that,
* train      = everything earlier (optionally only the last ``train_weeks``).

Any history used to score a split (model fit, features, popularity counts) must
come from rows strictly before that split's start: validation is scored with a
model fit on train, test with a model fit on train + validation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

DAY = pd.Timedelta(days=1)
WEEK = pd.Timedelta(days=7)


class LeakageError(AssertionError):
    """A split or a history frame contains rows it must not contain."""


@dataclass(frozen=True)
class SplitBoundaries:
    """Half-open day windows ``[start, end)``."""

    train_start: pd.Timestamp
    val_start: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    @property
    def train_cutoff(self) -> pd.Timestamp:
        """Exclusive upper bound for training rows (= validation start)."""
        return self.val_start

    def window(self, name: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        return {
            "train": (self.train_start, self.val_start),
            "validation": (self.val_start, self.test_start),
            "test": (self.test_start, self.test_end),
        }[name]

    def to_dict(self) -> dict[str, str]:
        return {k: v.date().isoformat() for k, v in asdict(self).items()}


@dataclass
class Split:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    boundaries: SplitBoundaries

    def target(self, name: str) -> pd.DataFrame:
        if name not in {"validation", "test"}:
            raise ValueError("target split must be 'validation' or 'test'")
        return getattr(self, name)

    def history_for(self, name: str) -> pd.DataFrame:
        """All rows strictly before the start of split ``name``."""
        if name == "validation":
            return self.train
        if name == "test":
            return pd.concat([self.train, self.validation], ignore_index=True)
        raise ValueError("history is defined for 'validation' or 'test'")


def make_boundaries(
    last_date: pd.Timestamp,
    val_weeks: int = 1,
    test_weeks: int = 1,
    train_weeks: int | None = None,
    first_date: pd.Timestamp | None = None,
) -> SplitBoundaries:
    if val_weeks < 1 or test_weeks < 1:
        raise ValueError("val_weeks and test_weeks must be >= 1")
    test_end = pd.Timestamp(last_date).normalize() + DAY
    test_start = test_end - test_weeks * WEEK
    val_start = test_start - val_weeks * WEEK
    if train_weeks is not None:
        train_start = val_start - train_weeks * WEEK
    elif first_date is not None:
        train_start = pd.Timestamp(first_date).normalize()
    else:
        train_start = pd.Timestamp("1900-01-01")
    return SplitBoundaries(train_start, val_start, test_start, test_end)


def _between(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df[(df["t_dat"] >= start) & (df["t_dat"] < end)].reset_index(drop=True)


def split_transactions(
    transactions: pd.DataFrame,
    val_weeks: int = 1,
    test_weeks: int = 1,
    train_weeks: int | None = None,
) -> Split:
    b = make_boundaries(
        transactions["t_dat"].max(),
        val_weeks,
        test_weeks,
        train_weeks,
        first_date=transactions["t_dat"].min(),
    )
    split = Split(
        train=_between(transactions, *b.window("train")),
        validation=_between(transactions, *b.window("validation")),
        test=_between(transactions, *b.window("test")),
        boundaries=b,
    )
    check_split(split)
    if split.train.empty or split.validation.empty or split.test.empty:
        raise ValueError(f"empty split: {b.to_dict()}")
    return split


# ----------------------------------------------------------------------------- leakage checks


def check_strictly_before(df: pd.DataFrame, start: pd.Timestamp, what: str = "history") -> None:
    """Raise if any row of ``df`` is on or after ``start``."""
    if len(df) and df["t_dat"].max() >= start:
        n = int((df["t_dat"] >= start).sum())
        raise LeakageError(f"{what}: {n} rows on or after {pd.Timestamp(start).date()}")


def check_no_train_after_cutoff(train: pd.DataFrame, cutoff: pd.Timestamp) -> None:
    check_strictly_before(train, cutoff, "train")


def check_within(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, what: str) -> None:
    if len(df) and (df["t_dat"].min() < start or df["t_dat"].max() >= end):
        raise LeakageError(f"{what}: rows outside [{start.date()}, {end.date()})")


def check_split(split: Split) -> None:
    """Train ends before validation starts; each target split stays inside its window."""
    b = split.boundaries
    check_no_train_after_cutoff(split.train, b.train_cutoff)
    check_within(split.validation, *b.window("validation"), "validation")
    check_within(split.test, *b.window("test"), "test")
    check_strictly_before(split.history_for("validation"), b.val_start, "validation history")
    check_strictly_before(split.history_for("test"), b.test_start, "test history")


def cold_start_report(history: pd.DataFrame, target: pd.DataFrame) -> dict[str, float]:
    """How many target users/items never appear in ``history`` (they need a fallback)."""
    users, items = target["customer_id"].unique(), target["article_id"].unique()
    seen_users = set(history["customer_id"].unique())
    seen_items = set(history["article_id"].unique())
    new_users = sum(u not in seen_users for u in users)
    new_items = sum(i not in seen_items for i in items)
    rows_new_item = int((~target["article_id"].isin(seen_items)).sum())
    return {
        "target_users": len(users),
        "cold_users": int(new_users),
        "cold_user_share": float(new_users / len(users)) if len(users) else 0.0,
        "target_items": len(items),
        "cold_items": int(new_items),
        "cold_item_share": float(new_items / len(items)) if len(items) else 0.0,
        "target_rows": len(target),
        "rows_with_cold_item": rows_new_item,
    }


def ground_truth(target: pd.DataFrame) -> dict[int, set[int]]:
    """Distinct articles each customer bought in the target window."""
    pairs = target[["customer_id", "article_id"]].drop_duplicates()
    return {int(u): set(map(int, g)) for u, g in pairs.groupby("customer_id")["article_id"]}

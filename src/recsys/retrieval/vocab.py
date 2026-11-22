"""Integer encodings for the two-tower model (index 0 = unknown / out of vocabulary)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from recsys.baselines.popularity import AGE_BINS

ITEM_ATTRS = (
    "product_type_no",
    "product_group_name",
    "colour_group_code",
    "department_no",
    "garment_group_no",
    "index_group_no",
)


class Vocab:
    """Maps values to 1..n; anything unseen maps to 0."""

    def __init__(self, values: Iterable) -> None:
        self.values = pd.Index(pd.unique(pd.Series(list(values)).dropna())).sort_values()

    def __len__(self) -> int:
        return len(self.values) + 1

    def encode(self, values: Sequence | np.ndarray | pd.Series) -> np.ndarray:
        idx = self.values.get_indexer(pd.Index(values))
        return np.where(idx >= 0, idx + 1, 0).astype(np.int64)

    def decode(self, idx: np.ndarray) -> np.ndarray:
        idx = np.asarray(idx)
        if (idx <= 0).any():
            raise ValueError("cannot decode the out-of-vocabulary index 0")
        return self.values.to_numpy()[idx - 1]

    def to_list(self) -> list:
        return [v.item() if hasattr(v, "item") else v for v in self.values.to_numpy()]


def age_bucket_index(ages: pd.Series, bins: Sequence[int] = AGE_BINS) -> np.ndarray:
    """1..len(bins)-1 for known ages, 0 for unknown."""
    b = pd.cut(ages, bins=list(bins), right=False, labels=False)
    return (b.fillna(-1).to_numpy() + 1).astype(np.int64)


class Catalogue:
    """Encoded attributes for every article in the articles table."""

    def __init__(self, articles: pd.DataFrame, vocabs: dict[str, Vocab] | None = None) -> None:
        self.vocabs = vocabs or {a: Vocab(articles[a].astype(str)) for a in ITEM_ATTRS}
        self.article_ids = pd.Index(articles["article_id"].to_numpy())
        self.attrs = np.stack(
            [self.vocabs[a].encode(articles[a].astype(str)) for a in ITEM_ATTRS], axis=1
        )

    def attr_sizes(self) -> list[int]:
        return [len(self.vocabs[a]) for a in ITEM_ATTRS]

    def encode(self, article_ids: Sequence[int] | np.ndarray) -> np.ndarray:
        """Attribute matrix (n x len(ITEM_ATTRS)); unknown articles get all zeros."""
        rows = self.article_ids.get_indexer(pd.Index(article_ids))
        out = np.zeros((len(rows), len(ITEM_ATTRS)), dtype=np.int64)
        known = rows >= 0
        out[known] = self.attrs[rows[known]]
        return out


def history_matrix(
    history: pd.DataFrame, users: Sequence[int], length: int, item_vocab: Vocab
) -> np.ndarray:
    """Each user's last ``length`` purchased items (encoded, 0-padded), oldest first."""
    users = pd.Index(users)
    out = np.zeros((len(users), length), dtype=np.int64)
    if history.empty or not len(users):
        return out
    sub = history[history["customer_id"].isin(users)]
    sub = (
        sub.sort_values(["customer_id", "t_dat"], kind="stable").groupby("customer_id").tail(length)
    )
    pos = sub.groupby("customer_id").cumcount().to_numpy()
    rows = users.get_indexer(sub["customer_id"])
    out[rows, pos] = item_vocab.encode(sub["article_id"])
    return out

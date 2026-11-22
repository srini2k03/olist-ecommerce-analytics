"""LightGBM LambdaRank over candidate lists (one group per user)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Self

import lightgbm as lgb
import numpy as np
import pandas as pd

from recsys.baselines.base import fill
from recsys.ranking.features import CATEGORICAL

BASE_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [12],
    "label_gain": [0, 1],  # binary relevance
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 4,
    "verbose": -1,
}


def _groups(df: pd.DataFrame) -> np.ndarray:
    """Group sizes in row order; rows must be sorted by customer_id."""
    if not df["customer_id"].is_monotonic_increasing:
        raise ValueError("rows must be sorted by customer_id")
    return df.groupby("customer_id", sort=False).size().to_numpy()


def with_positives(df: pd.DataFrame) -> pd.DataFrame:
    """Groups without a positive give LambdaRank no gradient; drop them for training."""
    has = df.groupby("customer_id")["label"].transform("max") > 0
    return df[has]


class LambdaRanker:
    def __init__(self, features: Sequence[str], params: dict | None = None, seed: int = 42) -> None:
        self.features = list(features)
        self.params = {**BASE_PARAMS, **(params or {}), "seed": seed}

    def _dataset(self, df: pd.DataFrame, reference=None) -> lgb.Dataset:
        cats = [c for c in CATEGORICAL if c in self.features]
        return lgb.Dataset(
            df[self.features],
            label=df["label"],
            group=_groups(df),
            categorical_feature=cats,
            reference=reference,
            free_raw_data=False,
        )

    def fit(
        self,
        train: pd.DataFrame,
        valid: pd.DataFrame | None = None,
        num_boost_round: int = 500,
        early_stopping: int | None = 50,
    ) -> Self:
        dtrain = self._dataset(with_positives(train))
        callbacks, valid_sets = [], []
        if valid is not None:
            valid_sets = [self._dataset(with_positives(valid), reference=dtrain)]
            if early_stopping:
                callbacks.append(lgb.early_stopping(early_stopping, verbose=False))
        self.booster_ = lgb.train(
            self.params, dtrain, num_boost_round, valid_sets=valid_sets, callbacks=callbacks
        )
        self.best_iteration_ = self.booster_.best_iteration or num_boost_round
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.booster_.predict(df[self.features], num_iteration=self.best_iteration_)

    def importances(self) -> pd.DataFrame:
        gain = self.booster_.feature_importance("gain", iteration=self.best_iteration_)
        out = pd.DataFrame({"feature": self.features, "gain": gain})
        out["gain_share"] = out["gain"] / out["gain"].sum()
        return out.sort_values("gain", ascending=False, ignore_index=True)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.booster_.save_model(str(path), num_iteration=self.best_iteration_)

    @classmethod
    def load(cls, path: Path) -> LambdaRanker:
        booster = lgb.Booster(model_file=str(path))
        obj = cls(booster.feature_name())
        obj.booster_ = booster
        obj.best_iteration_ = booster.current_iteration()
        return obj


def rank_candidates(
    df: pd.DataFrame, scores: np.ndarray, k: int, fallback: Sequence[int] = ()
) -> dict[int, list[int]]:
    """Top ``k`` articles per user by score (ties by article id), topped up from ``fallback``."""
    ranked = (
        df[["customer_id", "article_id"]]
        .assign(score=scores)
        .sort_values(["customer_id", "score", "article_id"], ascending=[True, False, True])
    )
    out = {}
    for u, g in ranked.groupby("customer_id", sort=False)["article_id"]:
        out[int(u)] = fill(g.head(k).astype(int).tolist(), fallback, k)
    return out

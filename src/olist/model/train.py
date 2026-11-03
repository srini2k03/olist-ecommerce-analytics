"""Model definitions: recency rule, logistic regression and LightGBM."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from olist.config import RANDOM_SEED
from olist.model.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

SKEWED = ["total_spend", "avg_order_value", "max_order_value", "n_items"]


def recency_rule_score(df: pd.DataFrame) -> np.ndarray:
    """Rule-of-thumb baseline: the more recently someone bought, the likelier they return."""
    return -df["recency_days"].to_numpy(float)


def make_logreg() -> Pipeline:
    numeric = [c for c in NUMERIC_FEATURES if c not in SKEWED]
    pre = ColumnTransformer(
        [
            (
                "skewed",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
                    StandardScaler(),
                ),
                SKEWED,
            ),
            (
                "num",
                make_pipeline(
                    SimpleImputer(strategy="median", add_indicator=True), StandardScaler()
                ),
                numeric,
            ),
            (
                "cat",
                make_pipeline(
                    SimpleImputer(strategy="constant", fill_value="missing"),
                    OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=300),
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline([("pre", pre), ("clf", LogisticRegression(C=0.5, max_iter=3000))])


def make_lgbm(**overrides) -> LGBMClassifier:
    params = dict(
        n_estimators=400,
        learning_rate=0.02,
        num_leaves=15,
        min_child_samples=200,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        cat_smooth=50,
        min_data_per_group=200,
        random_state=RANDOM_SEED,
        n_jobs=4,
        verbose=-1,
    )
    params.update(overrides)
    return LGBMClassifier(**params)


def to_lgbm_frame(df: pd.DataFrame, categories: dict[str, list[str]] | None = None):
    """Cast categoricals with a fixed category list so train and score frames agree."""
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    cats = categories or {
        c: sorted(X[c].dropna().astype(str).unique().tolist()) for c in CATEGORICAL_FEATURES
    }
    for c in CATEGORICAL_FEATURES:
        values = X[c].astype("string")
        values = values.where(values.isin(cats[c]))  # unseen categories -> missing
        X[c] = pd.Categorical(values, categories=cats[c])
    return X, cats

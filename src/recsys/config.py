"""Paths and constants for the recommender (all data paths are env-overridable)."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HM_DIR = Path(os.environ.get("RECSYS_DATA_DIR", REPO_ROOT / "data" / "hm"))
HM_RAW_DIR = Path(os.environ.get("RECSYS_RAW_DIR", HM_DIR / "raw"))
RESULTS_DIR = Path(os.environ.get("RECSYS_RESULTS_DIR", REPO_ROOT / "results" / "recsys"))

# Sub-directories of a data dir (HM_DIR for real data, any folder for synthetic runs).
PARQUET_SUBDIR = "parquet"
SUBSET_SUBDIR = "subset"
SPLIT_SUBDIR = "split"

COMPETITION = "h-and-m-personalized-fashion-recommendations"
RAW_FILES = ("transactions_train.csv", "articles.csv", "customers.csv")

RANDOM_SEED = 42
SUBSET_WEEKS = 16
SUBSET_CUSTOMER_FRAC = 0.25
EVAL_KS = (12, 50, 100)

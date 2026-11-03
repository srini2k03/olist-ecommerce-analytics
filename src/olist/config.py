"""Paths and constants shared by scripts, notebooks, tests and the app."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("OLIST_DATA_DIR", REPO_ROOT / "data"))
RAW_DIR = Path(os.environ.get("OLIST_RAW_DIR", DATA_DIR / "raw"))
WAREHOUSE_PATH = Path(os.environ.get("OLIST_DUCKDB", DATA_DIR / "olist.duckdb"))
SQL_DIR = REPO_ROOT / "sql"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
SQL_RESULTS_DIR = RESULTS_DIR / "sql"
APP_DATA_DIR = REPO_ROOT / "app" / "data"

# Orders in these states never turned into revenue.
EXCLUDED_ORDER_STATUSES = ("canceled", "unavailable")

# The months at both ends of the dataset are partial (the shop was ramping up
# in late 2016 and the extract stops mid-way through September/October 2018).
ANALYSIS_START = "2017-01-01"
ANALYSIS_END = "2018-09-01"  # exclusive

RANDOM_SEED = 42

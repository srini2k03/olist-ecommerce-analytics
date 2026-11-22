"""A recent, customer-sampled subset of H&M that fits comfortably in memory.

Keeps the last ``n_weeks`` of transactions and a seeded fraction of the
customers active in that window, then keeps every customer row the remaining
transactions reference. The articles table is kept whole: it is the product
catalogue, and filtering it by the kept transactions would leak which articles
sell in the validation and test weeks into the cold-start candidate pool. The
customer sample is a hash of the customer id and the seed, so the same customer
is in or out regardless of the window length.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from recsys.config import RANDOM_SEED, SUBSET_CUSTOMER_FRAC, SUBSET_WEEKS
from recsys.data.schema import HMData

log = logging.getLogger(__name__)


def _splitmix64(x: np.ndarray) -> np.ndarray:
    z = x + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def stable_uniform(ids: np.ndarray, seed: int = RANDOM_SEED) -> np.ndarray:
    """A deterministic pseudo-random number in [0, 1) per int64 id."""
    bits = np.ascontiguousarray(ids, dtype=np.int64).view(np.uint64)
    mixed = _splitmix64(bits ^ _splitmix64(np.array([seed], dtype=np.uint64)))
    return (mixed >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def last_n_weeks(transactions: pd.DataFrame, n_weeks: int) -> pd.DataFrame:
    """Rows in the final ``n_weeks * 7`` days, counted back from the last date."""
    if n_weeks < 1:
        raise ValueError("n_weeks must be >= 1")
    last = transactions["t_dat"].max()
    start = last - pd.Timedelta(days=7 * n_weeks - 1)
    return transactions[transactions["t_dat"] >= start]


def make_subset(
    data: HMData,
    n_weeks: int = SUBSET_WEEKS,
    customer_frac: float = SUBSET_CUSTOMER_FRAC,
    seed: int = RANDOM_SEED,
) -> HMData:
    if not 0 < customer_frac <= 1:
        raise ValueError("customer_frac must be in (0, 1]")
    tx = last_n_weeks(data.transactions, n_weeks)
    if customer_frac < 1:
        keep = stable_uniform(tx["customer_id"].to_numpy(), seed) < customer_frac
        tx = tx[keep]
    tx = tx.sort_values(["t_dat", "customer_id", "article_id"], kind="stable")
    articles = data.articles
    customers = data.customers[data.customers["customer_id"].isin(tx["customer_id"].unique())]
    out = HMData(tx, articles, customers).coerced()
    counts = out.counts()
    log.info(
        "subset: last %d weeks, %.0f%% of customers -> %s",
        n_weeks,
        100 * customer_frac,
        ", ".join(f"{k}={v:,}" for k, v in counts.items()),
    )
    missing_articles = int((~tx["article_id"].isin(out.articles["article_id"])).sum())
    if missing_articles:
        log.warning(
            "%d transactions reference articles missing from the catalogue", missing_articles
        )
    return out

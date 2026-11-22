"""Fixtures for the recommender tests: synthetic H&M-shaped data only, no network."""

from __future__ import annotations

import pytest

from recsys.data.split import split_transactions
from recsys.synthetic import generate


@pytest.fixture(scope="session")
def hm_data():
    return generate()


@pytest.fixture(scope="session")
def hm_split(hm_data):
    return split_transactions(hm_data.transactions)


@pytest.fixture(scope="session")
def synthetic_pipeline(tmp_path_factory):
    """The whole pipeline, run once in quick mode (needs torch, faiss and lightgbm)."""
    for mod in ("torch", "faiss", "lightgbm"):
        pytest.importorskip(mod)
    from recsys.cli import main

    data_dir = tmp_path_factory.mktemp("hm_synthetic")
    assert main(["--synthetic", "--quick", "--n-boot", "50", "--data-dir", str(data_dir)]) == 0
    return data_dir

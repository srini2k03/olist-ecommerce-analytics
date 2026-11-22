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

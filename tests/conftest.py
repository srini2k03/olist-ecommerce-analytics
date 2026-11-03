"""Shared fixtures: a synthetic warehouse (no network needed) and optional Postgres."""

from __future__ import annotations

import os

import pytest

from olist.db import DuckDBEngine, PostgresEngine, build_duckdb, build_postgres
from olist.synthetic import write_csvs


@pytest.fixture(scope="session")
def synthetic_raw(tmp_path_factory):
    return write_csvs(tmp_path_factory.mktemp("raw"))


@pytest.fixture(scope="session")
def synthetic_duckdb(synthetic_raw, tmp_path_factory):
    path = tmp_path_factory.mktemp("wh") / "olist.duckdb"
    build_duckdb(path, synthetic_raw)
    return path


@pytest.fixture(scope="session")
def duck(synthetic_duckdb):
    eng = DuckDBEngine(synthetic_duckdb)
    yield eng
    eng.close()


@pytest.fixture(scope="session")
def pg(synthetic_raw):
    """Postgres engine loaded with the synthetic data.

    Uses OLIST_PG_TEST_DSN (a database that may be overwritten), never the
    real-data database.
    """
    dsn = os.environ.get("OLIST_PG_TEST_DSN")
    if not dsn:
        pytest.skip("OLIST_PG_TEST_DSN not set")
    build_postgres(dsn, synthetic_raw)
    eng = PostgresEngine(dsn)
    yield eng
    eng.close()


@pytest.fixture(scope="session")
def synthetic_orders(synthetic_duckdb):
    from olist.warehouse import connect, order_facts

    con = connect(synthetic_duckdb)
    try:
        return order_facts(con)
    finally:
        con.close()

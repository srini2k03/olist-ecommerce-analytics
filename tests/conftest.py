"""Shared fixtures: synthetic and hand-made warehouses (no network needed), optional
Postgres. Every warehouse fixture runs the full layered pipeline (raw load, checks,
staging, marts)."""

from __future__ import annotations

import os

import mini_data
import pytest

from olist.db import DuckDBEngine, PostgresEngine, build_postgres
from olist.pipeline.run import build_layers, run_duckdb
from olist.synthetic import write_csvs


@pytest.fixture(scope="session")
def synthetic_raw(tmp_path_factory):
    return write_csvs(tmp_path_factory.mktemp("raw"))


@pytest.fixture(scope="session")
def synthetic_duckdb(synthetic_raw, tmp_path_factory):
    path = tmp_path_factory.mktemp("wh") / "olist.duckdb"
    run_duckdb(path, synthetic_raw, report_dir=None)
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
    build_layers(eng)
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


@pytest.fixture(scope="session")
def mini_raw(tmp_path_factory):
    return mini_data.write_csvs(tmp_path_factory.mktemp("mini_raw"))


@pytest.fixture(scope="session")
def mini_report(mini_raw, tmp_path_factory):
    path = tmp_path_factory.mktemp("mini_wh") / "olist.duckdb"
    report = run_duckdb(path, mini_raw, report_dir=None)
    return path, report


@pytest.fixture(scope="session")
def mini(mini_report):
    eng = DuckDBEngine(mini_report[0])
    yield eng
    eng.close()

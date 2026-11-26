"""The BigQuery translation layer (no Google Cloud access needed)."""

import sqlglot

from olist.bigquery import OVERRIDE_DIR, bigquery_sql, translate
from olist.sql_runner import list_queries


def test_every_query_has_valid_bigquery_sql():
    for q in list_queries():
        sql, how = bigquery_sql(q)
        assert how in {"hand-written", "translated"}
        sqlglot.parse_one(sql, read="bigquery")  # raises if it is not valid BigQuery SQL
        code = "\n".join(line for line in sql.splitlines() if not line.lstrip().startswith("--"))
        assert "WITHIN GROUP" not in code.upper(), q.name


def test_percentile_queries_are_hand_written():
    for q in list_queries():
        if "WITHIN GROUP" in q.sql.upper():
            assert (OVERRIDE_DIR / f"{q.name}.sql").exists(), q.name


def test_overrides_match_a_real_query():
    names = {q.name for q in list_queries()}
    assert {p.stem for p in OVERRIDE_DIR.glob("*.sql")} <= names


def test_day_of_week_is_shifted_to_zero_based_sunday():
    out = translate("SELECT EXTRACT(DOW FROM ts) AS d FROM t")
    assert "DAYOFWEEK" in out and "- 1" in out


def test_date_minus_date_becomes_date_diff():
    out = translate("SELECT CAST(a AS DATE) - CAST(b AS DATE) AS days FROM t")
    assert "DATE_DIFF" in out and "DAY" in out

"""Synthetic generator, schema, Kaggle fetch (mocked), parquet conversion and subset."""

from __future__ import annotations

import hashlib
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from recsys.config import COMPETITION, RAW_FILES
from recsys.data import fetch_hm
from recsys.data.schema import (
    ARTICLE_DTYPES,
    CUSTOMER_DTYPES,
    TRANSACTION_DTYPES,
    HMData,
    hash_customer_ids,
)
from recsys.data.subset import last_n_weeks, make_subset, stable_uniform
from recsys.synthetic import generate, generate_with_hex_ids, to_raw_csvs

# ----------------------------------------------------------------------------- synthetic


def _dtypes(df):
    return {c: str(t) for c, t in df.dtypes.items()}


def test_synthetic_has_hm_columns_and_dtypes(hm_data):
    assert _dtypes(hm_data.transactions) == TRANSACTION_DTYPES
    assert _dtypes(hm_data.articles) == ARTICLE_DTYPES
    assert _dtypes(hm_data.customers) == CUSTOMER_DTYPES
    assert len(hm_data.transactions) == 30_000
    assert len(hm_data.articles) == 1_000
    assert len(hm_data.customers) == 2_000


def test_synthetic_is_seeded():
    small = {"n_customers": 200, "n_articles": 100, "n_transactions": 2_000}
    a, b = generate(**small), generate(**small)
    pd.testing.assert_frame_equal(a.transactions, b.transactions)
    pd.testing.assert_frame_equal(a.customers, b.customers)
    c = generate(seed=7, **small)
    assert not a.transactions.equals(c.transactions)


def test_synthetic_has_the_structure_baselines_need(hm_data):
    tx, articles = hm_data.transactions, hm_data.articles
    # referential integrity
    assert tx.article_id.isin(articles.article_id).all()
    assert tx.customer_id.isin(hm_data.customers.customer_id).all()
    # power-law popularity: the top 10% of articles carry most purchases
    counts = tx.article_id.value_counts()
    assert counts.head(len(counts) // 10).sum() / len(tx) > 0.4
    # repeat purchases and multi-variant product codes
    assert tx.duplicated(["customer_id", "article_id"]).mean() > 0.1
    assert (articles.groupby("product_code").size() > 1).any()
    assert hm_data.customers.age.between(16, 90).mean() > 0.95
    # new users and new items appear only in the final week
    last_week = tx.t_dat > tx.t_dat.max() - pd.Timedelta(days=7)
    before = tx[~last_week]
    assert (~tx[last_week].customer_id.isin(before.customer_id)).any()
    assert (~tx[last_week].article_id.isin(before.article_id)).any()


# ----------------------------------------------------------------------------- schema + fetch


def test_hash_customer_ids_uses_last_16_hex_digits():
    ids = ["a" * 48 + "f" * 16, "b" * 48 + "0" * 15 + "1", "c" * 48 + "7" + "f" * 15]
    assert hash_customer_ids(ids).tolist() == [-1, 1, 2**63 - 1]
    assert hash_customer_ids([]).dtype == np.int64


def test_missing_credentials_give_a_clear_error(tmp_path):
    env = {"KAGGLE_CONFIG_DIR": str(tmp_path)}
    assert not fetch_hm.credentials_available(env)
    with pytest.raises(fetch_hm.MissingCredentialsError, match="KAGGLE_USERNAME"):
        fetch_hm.download(tmp_path / "raw", env=env, runner=pytest.fail)
    (tmp_path / "kaggle.json").write_text("{}")
    assert fetch_hm.credentials_available(env)
    assert fetch_hm.credentials_available({"KAGGLE_USERNAME": "u", "KAGGLE_KEY": "k"})


def test_download_fetches_only_the_three_tables_and_writes_a_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_hm.shutil, "which", lambda _: "/usr/bin/kaggle")
    calls = []

    def fake_kaggle(cmd, check):
        calls.append(cmd)
        name, out = cmd[cmd.index("-f") + 1], cmd[cmd.index("-p") + 1]
        with zipfile.ZipFile(f"{out}/{name}.zip", "w") as zf:  # Kaggle zips large files
            zf.writestr(name, f"col\n{name}\n")

    raw = tmp_path / "raw"
    env = {"KAGGLE_USERNAME": "u", "KAGGLE_KEY": "k"}
    manifest = fetch_hm.download(raw, runner=fake_kaggle, env=env)
    assert [c[c.index("-f") + 1] for c in calls] == list(RAW_FILES)
    assert all(c[1:4] == ["competitions", "download", COMPETITION] for c in calls)
    assert sorted(p.name for p in raw.iterdir()) == sorted([*RAW_FILES, "manifest.json"])
    on_disk = json.loads((raw / "manifest.json").read_text())
    assert on_disk["files"] == manifest["files"]
    for name in RAW_FILES:
        data = (raw / name).read_bytes()
        assert manifest["files"][name] == {
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    # Present files are not downloaded again unless forced.
    calls.clear()
    fetch_hm.download(raw, runner=fake_kaggle, env=env)
    assert calls == []


def test_convert_raw_csvs_round_trips_to_typed_parquet(tmp_path):
    data, hex_ids = generate_with_hex_ids(n_customers=300, n_articles=200, n_transactions=3_000)
    to_raw_csvs(data, hex_ids, tmp_path / "raw")
    counts = fetch_hm.convert(tmp_path / "raw", tmp_path / "parquet", chunk_rows=700)
    assert counts == {"customers": 300, "articles": 200, "transactions": 3_000}
    back = HMData.read(tmp_path / "parquet")
    for name in ("transactions", "articles", "customers"):
        pd.testing.assert_frame_equal(
            getattr(back, name), getattr(data, name), check_categorical=False
        )
    mapping = pd.read_parquet(tmp_path / "parquet" / "customer_id_map.parquet")
    assert dict(zip(mapping.customer_id, mapping.customer_id_hex, strict=True)) == hex_ids


# ----------------------------------------------------------------------------- subset


def test_last_n_weeks_keeps_exactly_7n_days(hm_data):
    tx = last_n_weeks(hm_data.transactions, 4)
    assert tx.t_dat.max() == hm_data.transactions.t_dat.max()
    assert (tx.t_dat.max() - tx.t_dat.min()).days == 27
    with pytest.raises(ValueError):
        last_n_weeks(hm_data.transactions, 0)


def test_stable_uniform_is_deterministic_and_seeded():
    ids = np.arange(-50_000, 50_000, dtype=np.int64)
    u = stable_uniform(ids, 1)
    assert ((u >= 0) & (u < 1)).all()
    np.testing.assert_array_equal(u, stable_uniform(ids, 1))
    assert not np.array_equal(u, stable_uniform(ids, 2))
    assert abs((u < 0.25).mean() - 0.25) < 0.01


def test_subset_samples_customers_and_keeps_referenced_rows(hm_data):
    sub = make_subset(hm_data, n_weeks=8, customer_frac=0.5, seed=3)
    tx = sub.transactions
    window = last_n_weeks(hm_data.transactions, 8)
    assert tx.t_dat.min() >= window.t_dat.min()
    share = tx.customer_id.nunique() / window.customer_id.nunique()
    assert 0.4 < share < 0.6
    # A sampled customer keeps all of their rows in the window.
    kept = window[window.customer_id.isin(tx.customer_id)]
    assert len(kept) == len(tx)
    assert set(sub.articles.article_id) == set(tx.article_id)
    assert set(sub.customers.customer_id) == set(tx.customer_id)
    again = make_subset(hm_data, n_weeks=8, customer_frac=0.5, seed=3)
    pd.testing.assert_frame_equal(tx, again.transactions)
    with pytest.raises(ValueError):
        make_subset(hm_data, customer_frac=0)


def test_hmdata_write_read_round_trip(hm_data, tmp_path):
    hm_data.write(tmp_path)
    back = HMData.read(tmp_path)
    pd.testing.assert_frame_equal(back.transactions, hm_data.transactions)
    with pytest.raises(FileNotFoundError):
        HMData.read(tmp_path / "missing")

"""Week-based split boundaries, leakage checks and cold-start counting."""

from __future__ import annotations

import pandas as pd
import pytest

from recsys.data.split import (
    LeakageError,
    Split,
    check_no_train_after_cutoff,
    check_split,
    check_strictly_before,
    cold_start_report,
    ground_truth,
    make_boundaries,
    split_transactions,
)

T = pd.Timestamp


def _tx(rows):
    return pd.DataFrame(rows, columns=["t_dat", "customer_id", "article_id"]).assign(
        t_dat=lambda d: pd.to_datetime(d.t_dat)
    )


def test_boundaries_count_back_whole_weeks_from_the_last_date():
    b = make_boundaries(T("2020-09-22"))
    assert b.test_end == T("2020-09-23")
    assert b.test_start == T("2020-09-16")
    assert b.val_start == T("2020-09-09")
    assert b.train_cutoff == b.val_start
    b2 = make_boundaries(T("2020-09-22"), val_weeks=2, test_weeks=1, train_weeks=4)
    assert b2.val_start == T("2020-09-02")
    assert b2.train_start == T("2020-08-05")
    with pytest.raises(ValueError):
        make_boundaries(T("2020-09-22"), val_weeks=0)


def test_split_windows_partition_the_data(hm_data, hm_split):
    s, b = hm_split, hm_split.boundaries
    assert s.train.t_dat.max() < b.val_start
    assert s.validation.t_dat.min() >= b.val_start and s.validation.t_dat.max() < b.test_start
    assert s.test.t_dat.min() >= b.test_start
    assert s.test.t_dat.max() == hm_data.transactions.t_dat.max()
    assert s.validation.t_dat.nunique() == 7 and s.test.t_dat.nunique() == 7
    assert len(s.train) + len(s.validation) + len(s.test) == len(hm_data.transactions)


def test_train_weeks_limits_training_history(hm_data):
    s = split_transactions(hm_data.transactions, train_weeks=3)
    assert s.train.t_dat.nunique() == 21
    assert s.train.t_dat.max() < s.boundaries.val_start


def test_history_for_a_split_is_strictly_before_its_start(hm_split):
    b = hm_split.boundaries
    val_hist, test_hist = hm_split.history_for("validation"), hm_split.history_for("test")
    assert val_hist.t_dat.max() < b.val_start
    assert test_hist.t_dat.max() < b.test_start
    assert len(test_hist) == len(hm_split.train) + len(hm_split.validation)
    with pytest.raises(ValueError):
        hm_split.history_for("train")


def test_leakage_checks_catch_future_rows(hm_split):
    b = hm_split.boundaries
    leaked_train = pd.concat([hm_split.train, hm_split.validation.head(1)])
    with pytest.raises(LeakageError, match="train: 1 rows"):
        check_no_train_after_cutoff(leaked_train, b.train_cutoff)
    # A row on the split's first day is already "not before" the split.
    on_start = hm_split.validation[hm_split.validation.t_dat == b.val_start].head(2)
    with pytest.raises(LeakageError):
        check_strictly_before(pd.concat([hm_split.train, on_start]), b.val_start)
    check_strictly_before(hm_split.train, b.val_start)  # no error
    check_strictly_before(hm_split.train.head(0), b.val_start)  # empty is fine
    bad = Split(leaked_train, hm_split.validation, hm_split.test, b)
    with pytest.raises(LeakageError):
        check_split(bad)
    bad_test = Split(hm_split.train, hm_split.validation, hm_split.validation, b)
    with pytest.raises(LeakageError, match="test"):
        check_split(bad_test)


def test_cold_start_report_counts_unseen_users_and_items():
    history = _tx([("2020-01-01", 1, 10), ("2020-01-02", 2, 11)])
    target = _tx([("2020-01-08", 2, 11), ("2020-01-08", 3, 12), ("2020-01-09", 3, 12)])
    r = cold_start_report(history, target)
    assert r["target_users"] == 2 and r["cold_users"] == 1 and r["cold_user_share"] == 0.5
    assert r["target_items"] == 2 and r["cold_items"] == 1
    assert r["rows_with_cold_item"] == 2 and r["target_rows"] == 3


def test_synthetic_test_week_has_cold_users_and_items(hm_split):
    r = cold_start_report(hm_split.history_for("test"), hm_split.test)
    assert r["cold_users"] > 0 and r["cold_items"] > 0
    assert 0 < r["cold_user_share"] < 0.5


def test_ground_truth_is_distinct_articles_per_user():
    target = _tx(
        [("2020-01-08", 1, 5), ("2020-01-09", 1, 5), ("2020-01-09", 1, 6), ("2020-01-09", 2, 7)]
    )
    assert ground_truth(target) == {1: {5, 6}, 2: {7}}

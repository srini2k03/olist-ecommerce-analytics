"""Baselines: output shape, determinism and behaviour on hand-made histories."""

from __future__ import annotations

import pandas as pd
import pytest

from recsys.baselines import (
    AgePopularityRecommender,
    ItemCooccurrenceRecommender,
    PopularityRecommender,
    RepurchaseRecommender,
)
from recsys.baselines.base import fill, popularity_ranking


def _tx(rows):
    df = pd.DataFrame(rows, columns=["t_dat", "customer_id", "article_id"])
    return df.assign(t_dat=pd.to_datetime(df.t_dat))


def _all_baselines(customers):
    return [
        PopularityRecommender(weeks=1),
        AgePopularityRecommender(customers, weeks=4),
        ItemCooccurrenceRecommender(),
        RepurchaseRecommender(),
    ]


@pytest.mark.parametrize("idx", range(4))
def test_baselines_return_k_distinct_known_items_deterministically(hm_data, hm_split, idx):
    train = hm_split.train
    users = sorted(hm_split.validation.customer_id.unique())[:200] + [123456789]  # + unknown
    a = _all_baselines(hm_data.customers)[idx].fit(train).recommend(users, 50)
    b = _all_baselines(hm_data.customers)[idx].fit(train).recommend(users, 50)
    assert a == b
    assert list(a) == users
    known = set(train.article_id)
    for recs in a.values():
        assert len(recs) == 50 and len(set(recs)) == 50
        assert set(recs) <= known


def test_recommend_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        RepurchaseRecommender().recommend([1], 5)


def test_fill_dedupes_and_excludes():
    assert fill([3, 3, 1], [1, 2, 4, 5], 4) == [3, 1, 2, 4]
    assert fill([3], [1, 2], 3, exclude=[1]) == [3, 2]
    assert fill([], [1], 5) == [1]


def test_popularity_counts_recent_weeks_and_breaks_ties_by_id():
    train = _tx(
        [
            ("2020-01-01", 1, 9), ("2020-01-01", 2, 9), ("2020-01-01", 3, 9),  # old, popular
            ("2020-01-20", 1, 5), ("2020-01-21", 2, 5),
            ("2020-01-21", 3, 7), ("2020-01-21", 4, 6),
        ]
    )  # fmt: skip
    assert popularity_ranking(train) == [9, 5, 6, 7]
    recs = PopularityRecommender(weeks=1).fit(train).recommend([1, 2], 4)
    assert recs[1] == [5, 6, 7, 9]  # last week first (ties by id), then all-time fill
    assert PopularityRecommender(weeks=None).fit(train).recommend([1], 2)[1] == [9, 5]


def test_age_popularity_uses_the_users_bucket_and_falls_back_for_unknown_age():
    customers = pd.DataFrame({"customer_id": [1, 2, 3, 4], "age": [18.0, 19.0, 45.0, None]})
    train = _tx(
        [("2020-01-01", 1, 100), ("2020-01-02", 2, 100), ("2020-01-02", 3, 200),
         ("2020-01-03", 3, 200), ("2020-01-03", 3, 201), ("2020-01-03", 4, 200)]
    )  # fmt: skip
    model = AgePopularityRecommender(customers, weeks=None, min_bucket_rows=1).fit(train)
    recs = model.recommend([1, 3, 4, 99], 3)
    assert recs[1] == [100, 200, 201]  # young bucket, filled with global
    assert recs[3] == [200, 201, 100]
    assert recs[4] == recs[99] == [200, 100, 201]  # global popularity


def test_cooccurrence_recommends_neighbours_within_the_window():
    train = _tx(
        [
            ("2020-01-01", 1, 10), ("2020-01-02", 1, 20),   # 10-20 within window
            ("2020-01-01", 2, 10), ("2020-01-03", 2, 20),   # 10-20 again
            ("2020-01-01", 3, 10), ("2020-01-04", 3, 30),   # 10-30 once
            ("2020-01-01", 4, 10), ("2020-01-30", 4, 40),   # 10-40 outside 7-day window
            ("2020-01-30", 5, 50), ("2020-01-30", 6, 50),   # popular fallback
            ("2020-01-10", 7, 10),
        ]
    )  # fmt: skip
    model = ItemCooccurrenceRecommender(window_days=7, pair_weeks=None).fit(train)
    assert dict(model.neighbours_[10]) == {20: 2, 30: 1}
    recs = model.recommend([7, 99], 4)
    assert recs[7][:2] == [20, 30]
    assert 10 not in recs[7]  # the seed itself is left to the repurchase baseline
    assert recs[7][2] == 50  # popularity fill
    assert recs[99][0] == 50  # unknown user: popularity


def test_repurchase_ranks_by_recency_then_frequency():
    train = _tx(
        [
            ("2019-12-01", 1, 14), ("2020-01-01", 1, 11), ("2020-01-05", 1, 12),
            ("2020-01-05", 1, 13), ("2020-01-03", 1, 13), ("2020-01-05", 2, 99),
            ("2020-01-05", 3, 99), ("2020-01-05", 4, 99),
        ]
    )  # fmt: skip
    recs = RepurchaseRecommender(weeks=None).fit(train).recommend([1, 42], 5)
    assert recs[1] == [13, 12, 11, 14, 99]  # 13 and 12 tie on date; 13 was bought twice
    assert recs[42][0] == 99  # no history: popularity
    recent = RepurchaseRecommender(weeks=1).fit(train).recommend([1], 4)
    assert recent[1] == [13, 12, 11, 99]  # the purchase from December is outside the window

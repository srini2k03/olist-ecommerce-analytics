"""Cold users (popularity/trending blend) and new articles (attribute-only retrieval)."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from recsys.coldstart.items import new_item_metrics, users_buying_new_items
from recsys.coldstart.users import ColdUserRecommender


def _tx(rows):
    df = pd.DataFrame(rows, columns=["t_dat", "customer_id", "article_id"])
    return df.assign(t_dat=pd.to_datetime(df.t_dat))


def test_cold_user_blend_hand_computed():
    customers = pd.DataFrame({"customer_id": [1, 2, 3], "age": [18.0, 19.0, 45.0]})
    rows = [("2020-01-02", 3, 50)] * 6  # popular two weeks ago, flat
    rows += [("2020-01-10", 3, 50)] * 6
    rows += [("2020-01-10", 1, 60)] * 2 + [("2020-01-11", 2, 60)]  # young, new and rising
    rows += [("2020-01-11", 3, 70)]
    train = _tx(rows)
    trending = ColdUserRecommender(customers, alpha=0.0, min_bucket_rows=1).fit(train)
    # trend: 50 -> 6 * 7 / 7 = 6; 60 -> 3 * 4 / 1 = 12; 70 -> 1 * 2 / 1 = 2
    assert trending.recommend_for_age(None, 3) == [60, 50, 70]
    by_age = ColdUserRecommender(customers, alpha=1.0, min_bucket_rows=1).fit(train)
    assert by_age.recommend_for_age(18, 1) == [60]  # the young bucket bought only 60
    assert by_age.recommend_for_age(45, 1) == [50]
    assert by_age.recommend_for_age(None, 1) == [50]  # unknown age: global popularity
    recs = by_age.recommend([1, 3, 99], 2)
    assert recs[1][0] == 60 and recs[3][0] == 50 and recs[99] == by_age.recommend_for_age(None, 2)


def test_new_item_metrics_hand_computed():
    truth = {1: {10, 20}, 2: {30}, 3: {40}}
    recs = {1: [20, 99, 10], 2: [30, 1], 3: [5, 6]}
    m = new_item_metrics(recs, truth, new_items={20, 30, 40}, k=2)
    assert m["new_item_purchases"] == 3
    assert m["new_item_recall"] == pytest.approx(2 / 3)
    assert m["new_item_share"] == pytest.approx(2 / 6)
    assert users_buying_new_items(truth, {20, 40}) == [1, 3]


def test_attribute_path_retrieves_unseen_articles(hm_data, hm_split):
    pytest.importorskip("torch")
    from recsys.coldstart.run import cold_users
    from recsys.data.folds import make_folds
    from recsys.retrieval.model import TwoTowerConfig
    from recsys.retrieval.retriever import TwoTowerRetriever

    test = make_folds(hm_split)["test"]
    assert 0 < len(cold_users(test)) < len(test.users)
    seen = set(test.history.article_id)
    truth = test.truth
    new_items = {i for rel in truth.values() for i in rel} - seen
    assert new_items
    cfg = TwoTowerConfig(epochs=4, batch_size=512, dim=32, item_id_dropout=0.2)
    results = {}
    for name, c in (("attribute_path", cfg), ("no_id_dropout", replace(cfg, item_id_dropout=0.0))):
        r = TwoTowerRetriever(hm_data.articles, hm_data.customers, c).fit(test.history)
        items = r.index_items(include_unseen=True)
        assert set(items) >= new_items
        ids, _ = r.topk(test.users, 50, items=items)
        recs = dict(zip(test.users, ids.tolist(), strict=True))
        results[name] = new_item_metrics(recs, truth, new_items, 50)["new_item_recall"]
    assert results["attribute_path"] > 0
    assert results["attribute_path"] >= results["no_id_dropout"]

"""Candidate union, point-in-time features, leakage guards and the LambdaRank ranker."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("lightgbm")

from recsys.data.folds import Fold, make_folds  # noqa: E402
from recsys.data.split import LeakageError  # noqa: E402
from recsys.ranking.candidates import SOURCE_COLUMNS, CandidateConfig, CandidateGenerator  # noqa: E402
from recsys.ranking.features import FEATURES, FeatureStore, assemble  # noqa: E402
from recsys.ranking.ranker import LambdaRanker, rank_candidates, with_positives  # noqa: E402
from recsys.ranking.run import build_fold_data  # noqa: E402
from recsys.retrieval.model import TwoTowerConfig  # noqa: E402
from recsys.retrieval.retriever import TwoTowerRetriever  # noqa: E402

CFG = TwoTowerConfig(epochs=2, batch_size=512, dim=16)
CANDS = CandidateConfig(n_two_tower=30, n_cooc=10, n_repurchase=10, n_age_pop=10)


@pytest.fixture(scope="module")
def folds(hm_split):
    return make_folds(hm_split)


@pytest.fixture(scope="module")
def fold_data(hm_data, folds):
    out = {}
    for name in ("ranker_train", "validation"):
        fold = folds[name]
        retr = TwoTowerRetriever(hm_data.articles, hm_data.customers, CFG).fit(fold.history)
        out[name] = build_fold_data(fold, retr, hm_data.articles, hm_data.customers, CANDS)
    return out


def test_folds_are_point_in_time(folds, hm_split):
    b = hm_split.boundaries
    rt = folds["ranker_train"]
    assert rt.end == b.val_start and rt.start == b.val_start - pd.Timedelta(days=7)
    for fold in folds.values():
        assert fold.history.t_dat.max() < fold.start
        assert fold.target.t_dat.min() >= fold.start and fold.target.t_dat.max() < fold.end
    with pytest.raises(LeakageError):
        Fold("bad", rt.start, rt.end, pd.concat([rt.history, rt.target]), rt.target)


def test_ranker_training_data_uses_only_rows_before_its_week(fold_data, hm_data, folds):
    """Candidates and features for the ranker-training week depend only on its history:
    rewriting the target week changes the labels and nothing else."""
    fd = fold_data["ranker_train"]
    fold = fd.fold
    assert fd.store.as_of == fold.start
    assert fd.generator.as_of_ == fold.start
    rng = np.random.default_rng(0)
    other = fold.target.copy()
    other["article_id"] = rng.permutation(other["article_id"].to_numpy())
    changed = Fold(fold.name, fold.start, fold.end, fold.history, other)
    fd2 = build_fold_data(changed, fd.retriever, hm_data.articles, hm_data.customers, CANDS)
    cols = ["customer_id", "article_id", *FEATURES]
    pd.testing.assert_frame_equal(fd.frame[cols], fd2.frame[cols])
    assert not fd.frame["label"].equals(fd2.frame["label"])
    with pytest.raises(LeakageError):
        FeatureStore.build(
            pd.concat([fold.history, fold.target]), hm_data.articles, hm_data.customers, fold.start
        )


def test_candidates_record_their_sources(fold_data):
    frame = fold_data["validation"].frame
    assert frame.groupby("customer_id").size().min() >= CANDS.n_two_tower
    assert not frame.duplicated(["customer_id", "article_id"]).any()
    assert (frame[SOURCE_COLUMNS].sum(axis=1) >= 1).all()
    flags = frame["sources"].str.split("|")
    assert (flags.str.len() == frame["n_sources"]).all()
    assert frame.loc[frame.src_two_tower == 1, "tt_rank"].notna().all()
    assert frame["tt_score"].notna().all()
    assert set(FEATURES) <= set(frame.columns)
    assert frame["label"].sum() > 0


def test_generator_pickle_round_trip(fold_data, tmp_path):
    fd = fold_data["validation"]
    fd.generator.save(tmp_path / "gen.pkl")
    gen = CandidateGenerator.load(tmp_path / "gen.pkl")
    users = fd.fold.users[:10]
    vecs = fd.retriever.user_vectors(users, fd.fold.history)
    ids, _ = fd.retriever.topk(users, CANDS.n_two_tower, fd.fold.history)
    a = gen.generate(users, vecs, ids)
    b = fd.generator.generate(users, vecs, ids)
    pd.testing.assert_frame_equal(a, b)


def test_history_features_hand_computed(hm_data):
    art = hm_data.articles.iloc[:2]
    a1, a2 = (int(x) for x in art.article_id)
    history = pd.DataFrame(
        {
            "t_dat": pd.to_datetime(["2020-01-01", "2020-01-05", "2020-01-08"]),
            "customer_id": [7, 7, 7],
            "article_id": [a1, a1, a2],
            "price": np.float32([0.01, 0.02, 0.03]),
            "sales_channel_id": np.int8([2, 1, 2]),
        }
    )
    customers = pd.DataFrame({"customer_id": [7], "age": [30.0]})
    store = FeatureStore.build(history, hm_data.articles, customers, pd.Timestamp("2020-01-10"))
    cands = pd.DataFrame(
        {"customer_id": [7, 7], "article_id": [a1, a2], "tt_score": 0.5, "tt_rank": [1, 2],
         "cooc_score": np.nan, "agepop_rank": np.nan, "src_two_tower": 1, "src_cooc": 0,
         "src_repurchase": [1, 1], "src_age_pop": 0}
    )  # fmt: skip
    f = assemble(cands, store).set_index("article_id")
    assert f.loc[a1, "ui_count"] == 2 and f.loc[a1, "ui_days_since"] == 5
    assert f.loc[a2, "ui_days_since"] == 2
    assert f.loc[a1, "u_frequency"] == 3 and f.loc[a1, "u_recency_days"] == 2
    assert f.loc[a1, "u_channel2_share"] == pytest.approx(2 / 3)
    assert f.loc[a1, "i_pop_1w"] == 1 and f.loc[a1, "i_pop_4w"] == 2
    assert f.loc[a1, "u_age"] == 30 and f.loc[a1, "n_sources"] == 2
    assert f.loc[a1, "cooc_score"] == 0


def test_ranker_trains_predicts_and_round_trips(fold_data, tmp_path):
    train, val = fold_data["ranker_train"].frame, fold_data["validation"].frame
    ranker = LambdaRanker(FEATURES, {"num_leaves": 7, "min_data_in_leaf": 10}).fit(
        train, val, 30, 10
    )
    scores = ranker.predict(val)
    assert scores.shape == (len(val),)
    imp = ranker.importances()
    assert set(imp.feature) == set(FEATURES) and imp.gain_share.sum() == pytest.approx(1)
    ranker.save(tmp_path / "r.txt")
    np.testing.assert_allclose(LambdaRanker.load(tmp_path / "r.txt").predict(val), scores)
    assert with_positives(train).groupby("customer_id")["label"].max().min() == 1
    with pytest.raises(ValueError, match="sorted"):
        ranker.fit(train.iloc[::-1], None, 5, None)


def test_rank_candidates_orders_by_score_and_fills():
    df = pd.DataFrame({"customer_id": [1, 1, 1, 2], "article_id": [10, 11, 12, 10]})
    recs = rank_candidates(df, np.array([0.1, 0.9, 0.9, 0.5]), 3, fallback=[99, 10])
    assert recs == {1: [11, 12, 10], 2: [10, 99]}

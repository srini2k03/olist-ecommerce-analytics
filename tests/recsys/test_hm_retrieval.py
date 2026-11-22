"""Two-tower retrieval and FAISS indexes on synthetic data (CPU, seconds)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("faiss")

from recsys.data.folds import make_folds  # noqa: E402
from recsys.evaluate.metrics import recall_at_k  # noqa: E402
from recsys.retrieval import index as faiss_index  # noqa: E402
from recsys.retrieval.model import TwoTowerConfig, in_batch_loss  # noqa: E402
from recsys.retrieval.retriever import TwoTowerRetriever  # noqa: E402
from recsys.retrieval.vocab import Vocab, age_bucket_index, history_matrix  # noqa: E402

CFG = TwoTowerConfig(epochs=4, patience=4, batch_size=512, dim=32)


@pytest.fixture(scope="module")
def folds(hm_split):
    return make_folds(hm_split)


@pytest.fixture(scope="module")
def retriever(hm_data, folds):
    val = folds["validation"]
    return TwoTowerRetriever(hm_data.articles, hm_data.customers, CFG).fit(val.history, val)


def test_vocab_and_history_matrix():
    v = Vocab([30, 10, 20, 10])
    assert len(v) == 4
    assert v.encode([10, 20, 30, 99]).tolist() == [1, 2, 3, 0]
    assert v.decode(np.array([3])).tolist() == [30]
    hist = pd.DataFrame(
        {
            "t_dat": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-01"]),
            "customer_id": [1, 1, 1, 2],
            "article_id": [10, 20, 30, 99],
        }
    )
    m = history_matrix(hist, [2, 1, 3], 2, v)
    assert m.tolist() == [[0, 0], [2, 3], [0, 0]]  # last 2 items, oldest first; unknown -> 0
    assert age_bucket_index(pd.Series([18.0, None, 70.0])).tolist() == [1, 0, 7]


def test_in_batch_loss_masks_duplicate_positives():
    u = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    items = torch.tensor([1, 1, 2, 3])
    loss = in_batch_loss(u, u.clone(), items, torch.zeros(4), 0.1)
    assert torch.isfinite(loss)


def test_training_loss_decreases_and_early_stopping_is_recorded(retriever):
    assert len(retriever.losses_) == len(retriever.val_recall_) >= 2
    assert retriever.losses_[-1] < retriever.losses_[0]
    assert 1 <= retriever.best_epoch_ <= CFG.epochs
    assert max(retriever.val_recall_) == pytest.approx(
        retriever.val_recall_[retriever.best_epoch_ - 1]
    )


def test_embedding_shapes_and_norms(retriever, folds):
    users = folds["validation"].users[:50] + [999]  # 999: unknown user (OOV path)
    u = retriever.user_vectors(users)
    items = retriever.index_items()
    v = retriever.item_vectors(items)
    assert u.shape == (51, CFG.dim) and v.shape == (len(items), CFG.dim)
    assert u.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(u, axis=1), 1, rtol=1e-5)
    with_unseen = retriever.index_items(include_unseen=True)
    assert len(with_unseen) > len(items)
    assert set(items) < set(with_unseen)


def test_recall_beats_random(retriever, folds):
    val = folds["validation"]
    recall = retriever.evaluate_recall(val, 100)
    rng = np.random.default_rng(0)
    items = retriever.index_items()
    random_recall = np.mean(
        [
            recall_at_k(rng.choice(items, 100, replace=False).tolist(), rel, 100)
            for rel in val.truth.values()
        ]
    )
    assert recall > 2 * random_recall


def test_training_is_seeded(hm_data, folds):
    val = folds["validation"]
    cfg = TwoTowerConfig(epochs=1, batch_size=512, dim=16)
    a = TwoTowerRetriever(hm_data.articles, hm_data.customers, cfg).fit(val.history)
    b = TwoTowerRetriever(hm_data.articles, hm_data.customers, cfg).fit(val.history)
    np.testing.assert_allclose(
        a.item_vectors([a.index_items()[0]]), b.item_vectors([b.index_items()[0]]), atol=1e-6
    )


def test_save_load_round_trip(retriever, hm_data, folds, tmp_path):
    retriever.save(tmp_path)
    val = folds["validation"]
    loaded = TwoTowerRetriever.load(tmp_path, hm_data.articles, hm_data.customers, val.history)
    users = val.users[:20]
    np.testing.assert_allclose(loaded.user_vectors(users), retriever.user_vectors(users), atol=1e-6)
    assert loaded.best_epoch_ == retriever.best_epoch_


def test_faiss_exact_matches_torch_and_hnsw_is_close(retriever, folds):
    users = folds["validation"].users
    items = retriever.index_items()
    v, u = retriever.item_vectors(items), retriever.user_vectors(users)
    flat, hnsw = faiss_index.build_flat(v), faiss_index.build_hnsw(v)
    _, exact = faiss_index.search(flat, u, 50)
    torch_ids, _ = retriever.topk(users, 50)
    assert np.mean(items[exact] == torch_ids) > 0.99  # ties may swap places
    _, approx = faiss_index.search(hnsw, u, 50)
    assert faiss_index.overlap_at_k(exact, approx) > 0.9
    lat = faiss_index.latency_ms(hnsw, u, 50, n=20)
    assert lat["n"] == 20 and lat["p50"] <= lat["p95"] <= lat["p99"]

"""Serving bundle, service paths, FastAPI endpoints, benchmark and the demo UI."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")


@pytest.fixture(scope="module")
def bundle(synthetic_pipeline):
    return synthetic_pipeline / "artifacts" / "serving"


@pytest.fixture(scope="module")
def client(bundle):
    from fastapi.testclient import TestClient

    from recsys.serve.api import create_app

    with TestClient(create_app(bundle)) as c:
        yield c


@pytest.fixture(scope="module")
def service(bundle):
    from recsys.serve.service import RecommenderService

    return RecommenderService(bundle)


def test_bundle_contents(bundle):
    for name in (
        "manifest.json",
        "user_ids.npy",
        "user_vectors.npy",
        "items_hnsw.faiss",
        "candidates.pkl",
        "ranker.txt",
        "cold_users.pkl",
        "articles.parquet",
        "features/users.parquet",
        "retriever/model.pt",
    ):
        assert (bundle / name).exists(), name


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["data"] == "synthetic" and body["torch"] is True


def test_recommend_known_customer(client):
    cid = client.get("/users/sample", params={"n": 1}).json()["customer_ids"][0]
    body = client.get(f"/recommend/{cid}", params={"k": 12}).json()
    assert body["customer_id"] == cid and body["strategy"] == "two_tower+ranker"
    items = body["items"]
    assert len(items) == 12 and len({i["article_id"] for i in items}) == 12
    scores = [i["score"] for i in items if i["score"] is not None]
    assert scores == sorted(scores, reverse=True)
    assert all(i["sources"] for i in items) and "product_type_name" in items[0]
    history = client.get(f"/users/{cid}/history").json()["purchases"]
    assert history and {"t_dat", "article_id", "prod_name"} <= set(history[0])


def test_recommend_unknown_customer_uses_cold_start(client):
    body = client.get("/recommend/123", params={"k": 5, "age": 22}).json()
    assert body["strategy"].startswith("cold")
    assert len(body["items"]) == 5
    assert client.get("/users/123/history").json()["purchases"] == []


def test_recommend_validation_errors(client):
    assert client.get("/recommend/123", params={"k": 0}).status_code == 422
    assert client.get("/recommend/123", params={"k": 101}).status_code == 422
    assert client.get("/recommend/not-an-id").status_code == 400


def test_items_endpoint(client, service):
    article = int(service.items[0])
    body = client.get(f"/items/{article}").json()
    assert body["article_id"] == article and "colour_group_name" in body
    assert client.get("/items/1").status_code == 404


def test_service_paths(service):
    from recsys.data.schema import hash_customer_ids
    from recsys.serve.service import parse_customer_id

    hex_id = "ab" * 24 + "0" * 15 + "7"
    assert parse_customer_id(hex_id) == int(hash_customer_ids([hex_id])[0]) == 7
    assert parse_customer_id("42") == 42
    warm = int(service.user_ids[0])
    assert service.recommend(warm, 8)["strategy"] == "two_tower+ranker"
    # A customer with history but no stored vector gets one from the user tower.
    service._user_row = service._user_row.drop(warm)
    try:
        assert service.recommend(warm, 8)["strategy"] == "two_tower(on the fly)+ranker"
    finally:
        service._user_row = service._user_row.insert(0, warm)


def test_benchmark(bundle):
    from recsys.serve.benchmark import bench_in_process

    res = bench_in_process(bundle, n=30, warmup=2)
    assert res["requests"] == 30 and res["p50_ms"] <= res["p95_ms"] <= res["p99_ms"]
    assert sum(res["strategies"].values()) == 30


def test_streamlit_demo_against_the_api(client, monkeypatch):
    pytest.importorskip("streamlit")
    from pathlib import Path

    import requests
    from streamlit.testing.v1 import AppTest

    def fake_get(url, params=None, timeout=None):
        return client.get(url.removeprefix("http://localhost:8000"), params=params)

    monkeypatch.setattr(requests, "get", fake_get)
    app = Path(__file__).resolve().parents[2] / "app_recsys" / "streamlit_app.py"
    at = AppTest.from_file(str(app), default_timeout=60).run()
    assert not at.exception, at.exception
    assert at.title[0].value == "H&M recommender demo"
    assert any("SYNTHETIC" in w.value for w in at.warning)
    assert len(at.dataframe) >= 1
    at.sidebar.selectbox[0].set_value(at.sidebar.selectbox[0].options[1]).run()
    assert not at.exception and len(at.dataframe) == 2  # history + recommendations

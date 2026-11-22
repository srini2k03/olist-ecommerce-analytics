"""FastAPI service: ``uvicorn recsys.serve.api:app``.

Reads the serving bundle from ``$RECSYS_ARTIFACTS`` (default
``data/hm/artifacts/serving``) once at startup. Customer ids are returned as
strings because the int64 ids exceed JavaScript's safe-integer range.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request

from recsys import config
from recsys.serve.service import RecommenderService, parse_customer_id

DEFAULT_BUNDLE = config.HM_DIR / "artifacts" / "serving"


def create_app(folder: Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        path = folder or Path(os.environ.get("RECSYS_ARTIFACTS", DEFAULT_BUNDLE))
        app.state.service = RecommenderService(path)
        yield

    app = FastAPI(title="H&M recommender", version="0.1.0", lifespan=lifespan)

    def service(request: Request) -> RecommenderService:
        return request.app.state.service

    def customer_or_400(raw: str) -> str:
        try:
            parse_customer_id(raw)
        except ValueError as exc:
            raise HTTPException(400, "customer_id must be an integer or a 64-char hex id") from exc
        return raw

    @app.get("/health")
    def health(request: Request) -> dict:
        svc = service(request)
        m = svc.manifest
        return {
            "status": "ok",
            "data": m["data"],
            "as_of": m["as_of"],
            "users_with_embeddings": m["users_with_embeddings"],
            "index_items": m["index_items"],
            "cold_user_path": svc.cold_path,
            "torch": svc.retriever is not None,
        }

    @app.get("/recommend/{customer_id}")
    def recommend(
        request: Request,
        customer_id: str,
        k: int = Query(12, ge=1, le=100),
        age: float | None = Query(None, ge=0, le=120, description="for new customers"),
    ) -> dict:
        return service(request).recommend(customer_or_400(customer_id), k, age)

    @app.get("/items/{article_id}")
    def item(request: Request, article_id: int) -> dict:
        found = service(request).item(article_id)
        if found is None:
            raise HTTPException(404, f"article {article_id} not found")
        return found

    @app.get("/users/sample")
    def sample_users(request: Request, n: int = Query(10, ge=1, le=100), seed: int = 0) -> dict:
        return {"customer_ids": service(request).sample_users(n, seed)}

    @app.get("/users/{customer_id}/history")
    def history(request: Request, customer_id: str) -> dict:
        rows = service(request).user_history(customer_or_400(customer_id))
        return {"customer_id": customer_id, "purchases": rows}

    return app


app = create_app()

"""FAISS indexes over item embeddings: exact for evaluation, HNSW for serving."""

from __future__ import annotations

import time
from pathlib import Path

import faiss
import numpy as np


def _f32(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.float32)


def build_flat(vectors: np.ndarray) -> faiss.Index:
    """Exact maximum inner product (= cosine, the vectors are L2-normalised)."""
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(_f32(vectors))
    return index


def build_hnsw(
    vectors: np.ndarray, m: int = 32, ef_construction: int = 200, ef_search: int = 128
) -> faiss.Index:
    index = faiss.IndexHNSWFlat(vectors.shape[1], m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search
    index.add(_f32(vectors))
    return index


def search(index: faiss.Index, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """(scores, positions) of the top ``k`` per query; positions are -1 when fewer exist."""
    return index.search(_f32(queries), min(k, index.ntotal))


def overlap_at_k(exact: np.ndarray, approx: np.ndarray) -> float:
    """Mean share of the exact top-k that the approximate search also returns."""
    if not len(exact):
        return 0.0
    k = exact.shape[1]
    return float(np.mean([len(set(e) & set(a)) / k for e, a in zip(exact, approx, strict=True)]))


def latency_ms(index: faiss.Index, queries: np.ndarray, k: int, n: int = 1000) -> dict[str, float]:
    """Single-query latency percentiles (one query at a time, as a service would)."""
    q = _f32(queries)
    times = []
    for i in range(min(n, len(q)) or 0):
        t0 = time.perf_counter()
        index.search(q[i : i + 1], k)
        times.append((time.perf_counter() - t0) * 1000)
    if not times:
        return {"p50": float("nan"), "p95": float("nan"), "p99": float("nan"), "n": 0}
    p50, p95, p99 = np.percentile(times, [50, 95, 99])
    return {"p50": float(p50), "p95": float(p95), "p99": float(p99), "n": len(times)}


def save(index: faiss.Index, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load(path: Path) -> faiss.Index:
    return faiss.read_index(str(path))

"""Latency benchmark: ``python -m recsys.serve.benchmark [--url URL] [--n 1000]``.

Without ``--url`` it times the in-process service (model + features + ranking,
no HTTP); with ``--url`` it times real HTTP requests against a running API.
Requests mix known customers and unknown ids (the cold-start path).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from recsys import config

COLD_SHARE = 0.1


def request_ids(user_ids: list[str], n: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    warm = rng.choice(user_ids, n)
    cold = rng.integers(1, 2**62, n).astype(str)  # random ids: almost surely unknown
    return [c if rng.random() < COLD_SHARE else w for w, c in zip(warm, cold, strict=True)]


def summarise(times_ms: list[float], strategies: list[str], mode: str, k: int) -> dict:
    p50, p95, p99 = np.percentile(times_ms, [50, 95, 99])
    return {
        "mode": mode,
        "requests": len(times_ms),
        "k": k,
        "p50_ms": float(p50),
        "p95_ms": float(p95),
        "p99_ms": float(p99),
        "mean_ms": float(np.mean(times_ms)),
        "strategies": {s: strategies.count(s) for s in sorted(set(strategies))},
    }


def bench_in_process(folder: Path, n: int, k: int = 12, seed: int = 0, warmup: int = 20) -> dict:
    from recsys.serve.service import RecommenderService

    svc = RecommenderService(folder)
    ids = request_ids(svc.sample_users(1000, seed), n + warmup, seed)
    times, strategies = [], []
    for i, cid in enumerate(ids):
        t0 = time.perf_counter()
        out = svc.recommend(cid, k)
        if i >= warmup:
            times.append((time.perf_counter() - t0) * 1000)
            strategies.append(out["strategy"])
    return summarise(times, strategies, "in_process", k) | {"data": svc.manifest["data"]}


def bench_http(url: str, n: int, k: int = 12, seed: int = 0, warmup: int = 20) -> dict:
    import httpx

    with httpx.Client(base_url=url, timeout=30) as client:
        users = client.get("/users/sample", params={"n": 100, "seed": seed}).json()["customer_ids"]
        data = client.get("/health").json()["data"]
        times, strategies = [], []
        for i, cid in enumerate(request_ids(users, n + warmup, seed)):
            t0 = time.perf_counter()
            r = client.get(f"/recommend/{cid}", params={"k": k})
            r.raise_for_status()
            if i >= warmup:
                times.append((time.perf_counter() - t0) * 1000)
                strategies.append(r.json()["strategy"])
    return summarise(times, strategies, "http", k) | {"data": data}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--artifacts", type=Path, default=config.HM_DIR / "artifacts" / "serving")
    p.add_argument("--url", help="benchmark a running API instead of the in-process service")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--k", type=int, default=12)
    p.add_argument("--out", type=Path, help="write the JSON summary here")
    args = p.parse_args(argv)
    res = (
        bench_http(args.url, args.n, args.k)
        if args.url
        else bench_in_process(args.artifacts, args.n, args.k)
    )
    text = json.dumps(res, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

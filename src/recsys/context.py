"""Shared state for the pipeline stages: paths, data source, split, folds and outputs."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cached_property
from pathlib import Path

import pandas as pd

from recsys import config
from recsys.data.folds import Fold, make_folds
from recsys.data.schema import HMData, coerce, write_json
from recsys.data.split import Split, SplitBoundaries, check_split

log = logging.getLogger(__name__)

SYNTHETIC_MARKER = "SYNTHETIC"
SPLIT_NAMES = ("train", "validation", "test")


def load_split(data_dir: Path) -> Split:
    folder = data_dir / config.SPLIT_SUBDIR
    report = json.loads((folder / "split_report.json").read_text())
    b = SplitBoundaries(**{k: pd.Timestamp(v) for k, v in report["boundaries"].items()})
    frames = {
        n: coerce(pd.read_parquet(folder / f"{n}.parquet"), "transactions") for n in SPLIT_NAMES
    }
    split = Split(boundaries=b, **frames)
    check_split(split)
    return split


def records(df: pd.DataFrame) -> list[dict]:
    return df.astype(object).where(df.notna(), None).to_dict(orient="records")


@dataclass
class Context:
    data_dir: Path
    results_dir: Path
    seed: int = config.RANDOM_SEED
    n_boot: int = 1000
    quick: bool = False  # smaller models and grids (tests, CI smoke runs)

    @property
    def source(self) -> str:
        return "synthetic" if (self.data_dir / SYNTHETIC_MARKER).exists() else "real"

    @property
    def artifacts_dir(self) -> Path:
        return self.data_dir / "artifacts"

    @cached_property
    def data(self) -> HMData:
        return HMData.read(self.data_dir / config.SUBSET_SUBDIR)

    @cached_property
    def split(self) -> Split:
        return load_split(self.data_dir)

    @cached_property
    def folds(self) -> dict[str, Fold]:
        return make_folds(self.split)

    def write_result(self, name: str, payload: dict) -> Path:
        path = self.results_dir / f"{name}.json"
        header = {
            "data": self.source,
            "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "boundaries": self.split.boundaries.to_dict(),
            "bootstrap": {"n_boot": self.n_boot, "alpha": 0.05, "seed": self.seed, "unit": "user"},
        }
        write_json(path, {**header, **payload})
        log.info("wrote %s", path)
        return path

    def save_recs(self, fold: str, stage: str, recs: dict[int, list[int]]) -> None:
        """Persist a stage's ranked recommendations for the final report."""
        rows = [(u, r, int(i)) for u, items in recs.items() for r, i in enumerate(items)]
        df = pd.DataFrame(rows, columns=["customer_id", "rank", "article_id"])
        folder = self.artifacts_dir / "recs" / fold
        folder.mkdir(parents=True, exist_ok=True)
        df.to_parquet(folder / f"{stage}.parquet", index=False)

    def load_recs(self, fold: str) -> dict[str, dict[int, list[int]]]:
        out = {}
        for path in sorted((self.artifacts_dir / "recs" / fold).glob("*.parquet")):
            df = pd.read_parquet(path).sort_values(["customer_id", "rank"])
            out[path.stem] = {
                int(u): g.tolist() for u, g in df.groupby("customer_id")["article_id"]
            }
        return out


def make_context(
    data_dir: Path,
    results_dir: Path | None = None,
    seed: int = config.RANDOM_SEED,
    n_boot: int = 1000,
    quick: bool = False,
) -> Context:
    """Synthetic data never writes into the committed ``results/recsys/`` folder."""
    synthetic = (data_dir / SYNTHETIC_MARKER).exists()
    if results_dir is None:
        results_dir = data_dir / "results" if synthetic else config.RESULTS_DIR
    elif synthetic and results_dir.resolve() == config.RESULTS_DIR.resolve():
        log.warning(
            "synthetic data: writing results to %s, not %s", data_dir / "results", results_dir
        )
        results_dir = data_dir / "results"
    return Context(data_dir, results_dir, seed, n_boot, quick)

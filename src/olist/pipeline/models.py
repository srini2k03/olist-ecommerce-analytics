"""SQL models of the staging and marts layers.

Each model is one ``SELECT`` in ``sql/staging/`` or ``sql/marts/`` whose header declares
its dependencies (``-- Depends on: a, b``). The builder sorts models topologically and
materialises each as a table with ``CREATE TABLE <name> AS <select>``, which DuckDB and
Postgres both support, so the same files build either engine.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from graphlib import TopologicalSorter
from pathlib import Path

from olist.config import SQL_DIR
from olist.db import Engine
from olist.logging_utils import get_logger
from olist.schema import TABLES_BY_NAME

LAYERS = ("staging", "marts")
log = get_logger(__name__)


@dataclass(frozen=True)
class Model:
    path: Path

    @property
    def name(self) -> str:
        return self.path.stem

    @property
    def layer(self) -> str:
        return self.path.parent.name

    @property
    def sql(self) -> str:
        return self.path.read_text(encoding="utf-8")

    @property
    def header(self) -> dict[str, str]:
        """``-- Key: value`` lines at the top of the file (continuations are joined)."""
        fields: dict[str, str] = {}
        key = None
        for line in self.sql.splitlines():
            if not line.startswith("--"):
                break
            text = line[2:].strip()
            m = re.match(r"^([A-Z][A-Za-z ()]+):\s*(.*)$", text)
            if m:
                key = m.group(1)
                fields[key] = m.group(2)
            elif key:
                fields[key] += " " + text
        return fields

    @property
    def depends_on(self) -> list[str]:
        raw = self.header.get("Depends on", "")
        deps = [d.strip() for d in raw.split(",") if d.strip()]
        bad = [d for d in deps if not re.fullmatch(r"[a-z_]+", d)]
        if bad:
            raise ValueError(f"{self.name}: malformed 'Depends on' entries {bad}")
        return deps

    @property
    def select(self) -> str:
        """The SQL without its comment header (and without a trailing semicolon)."""
        body = "\n".join(ln for ln in self.sql.splitlines() if not ln.startswith("--"))
        return body.strip().rstrip(";")


def list_models(sql_dir: Path = SQL_DIR) -> list[Model]:
    return [Model(p) for layer in LAYERS for p in sorted((Path(sql_dir) / layer).glob("*.sql"))]


def build_order(models: list[Model]) -> list[Model]:
    """Topological order; raises on unknown dependencies or cycles."""
    by_name = {m.name: m for m in models}
    graph: dict[str, set[str]] = {}
    for m in models:
        deps = set()
        for d in m.depends_on:
            if d in by_name:
                deps.add(d)
            elif d not in TABLES_BY_NAME:
                raise ValueError(f"{m.name} depends on unknown table {d!r}")
        graph[m.name] = deps
    return [by_name[n] for n in TopologicalSorter(graph).static_order()]


def build_models(engine: Engine, models: list[Model] | None = None) -> dict[str, dict]:
    """Materialise every model as a table. Returns {model: {rows, seconds, layer}}."""
    stats: dict[str, dict] = {}
    for m in build_order(models or list_models()):
        t0 = time.perf_counter()
        engine.execute(f"DROP TABLE IF EXISTS {m.name}")
        engine.execute(f"CREATE TABLE {m.name} AS\n{m.select}")
        if engine.name == "postgres":
            # Fresh CTAS tables have no planner statistics; without them Postgres picks
            # nested loops for the mart joins (fct_orders took minutes instead of seconds).
            engine.execute(f"ANALYZE {m.name}")
        rows = int(engine.query(f"SELECT COUNT(*) AS n FROM {m.name}")["n"].iloc[0])
        secs = time.perf_counter() - t0
        stats[m.name] = {"layer": m.layer, "rows": rows, "seconds": round(secs, 3)}
        log.info("model built", extra={"fields": {"model": m.name, "rows": rows, "s": secs}})
    return stats

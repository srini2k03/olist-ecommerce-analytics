"""Build the layered warehouse: data/olist.duckdb (and optionally Postgres) from data/raw.

Thin wrapper around ``python -m olist.pipeline``: typed raw load, data-quality and
referential-integrity checks, staging and marts, and the report in results/data_quality/.
"""

from __future__ import annotations

import sys

from olist.pipeline.run import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

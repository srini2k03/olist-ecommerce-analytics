"""Run every analysis on the warehouse marts and write results/analytics/ (see olist.analysis)."""

from __future__ import annotations

import json

from olist.analysis import run_all

if __name__ == "__main__":
    print(json.dumps(run_all(), indent=2))

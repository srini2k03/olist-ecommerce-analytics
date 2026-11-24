"""Olist e-commerce analytics dashboard (10 pages).

Runs with `streamlit run app/streamlit_app.py`, on Streamlit Community Cloud, and in the
browser on GitHub Pages via stlite. It reads only the prebuilt bundle in app/data and the
pure-Python parts of the `olist` package (metric layer, analytics, experiments, scenarios):
from src/ in the repository, or copied next to this file in the stlite build.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for candidate in (HERE, HERE.parent / "src"):
    if (candidate / "olist").is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from dashboard.app import main  # noqa: E402

main()

"""Build the GitHub Pages site: executed notebooks as HTML, the executive
summary, the README and its figures. Output goes to site/."""

from __future__ import annotations

import shutil
from pathlib import Path

import markdown
import nbformat
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "site"
NOTEBOOKS = ["01_eda", "02_stats", "03_segmentation", "04_churn"]
TITLES = {
    "01_eda": "01 · Exploratory analysis",
    "02_stats": "02 · Statistics and the simulated A/B test",
    "03_segmentation": "03 · RFM + K-Means segmentation",
    "04_churn": "04 · Repeat-purchase / churn model",
}

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{ --ink:#0b0b0b; --muted:#52514e; --bg:#fcfcfb; --line:#e4e3df; --link:#2a78d6; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --ink:#f0efec; --muted:#c3c2b7; --bg:#1a1a19; --line:#3a3a38; --link:#6da7ec; }}
  img {{ background:#fcfcfb; }}
}}
body {{ font: 16px/1.6 system-ui, -apple-system, Segoe UI, sans-serif; color:var(--ink);
       background:var(--bg); max-width: 920px; margin: 0 auto; padding: 24px 16px 64px; }}
a {{ color: var(--link); }}
img {{ max-width: 100%; height: auto; }}
table {{ border-collapse: collapse; display:block; overflow-x:auto; }}
th, td {{ border-bottom: 1px solid var(--line); padding: 4px 10px; text-align: left; }}
code {{ font-size: 0.92em; }}
nav {{ margin-bottom: 24px; color: var(--muted); }}
</style></head><body>
<nav><a href="index.html">Home</a> · <a href="executive_summary.html">Executive summary</a> ·
{nav}</nav>
{body}
</body></html>"""


def md_to_html(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    return markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "notebooks").mkdir(parents=True)
    shutil.copytree(ROOT / "results" / "figures", OUT / "results" / "figures")
    nav = " · ".join(f'<a href="notebooks/{n}.html">{n}</a>' for n in NOTEBOOKS)

    exporter = HTMLExporter(template_name="lab")
    for name in NOTEBOOKS:
        nb = nbformat.read(ROOT / "notebooks" / f"{name}.ipynb", as_version=4)
        body, _ = exporter.from_notebook_node(nb, resources={"metadata": {"name": TITLES[name]}})
        (OUT / "notebooks" / f"{name}.html").write_text(body, encoding="utf-8")

    readme = md_to_html(ROOT / "README.md").replace('href="notebooks/', 'href="notebooks/')
    for name in NOTEBOOKS:
        readme = readme.replace(f"notebooks/{name}.ipynb", f"notebooks/{name}.html")
    readme = readme.replace('href="EXECUTIVE_SUMMARY.md"', 'href="executive_summary.html"')
    (OUT / "index.html").write_text(
        PAGE.format(title="Olist e-commerce analytics", nav=nav, body=readme), encoding="utf-8"
    )
    summary = md_to_html(ROOT / "EXECUTIVE_SUMMARY.md")
    for name in NOTEBOOKS:
        summary = summary.replace(f"notebooks/{name}.ipynb", f"notebooks/{name}.html")
    (OUT / "executive_summary.html").write_text(
        PAGE.format(title="Executive summary", nav=nav, body=summary), encoding="utf-8"
    )
    (OUT / ".nojekyll").write_text("")
    print(f"site built in {OUT}: {sorted(p.name for p in OUT.iterdir())}")


if __name__ == "__main__":
    main()

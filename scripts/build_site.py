"""Build the GitHub Pages site into site/: a landing page with the key findings, the
executive summary as a one-pager, the four executed notebooks wrapped in the site chrome,
and the in-browser Streamlit dashboard (stlite) under site/app/.

The pages use the portfolio's design kit (web/sb-kit.css and web/sb-kit.js, copied unchanged
from https://srini2k03.github.io/) plus web/site.css. Every number comes from committed files:
README.md, EXECUTIVE_SUMMARY.md, results/metrics/*.json and results/sql/*.csv.
"""

from __future__ import annotations

import csv
import html
import json
import posixpath
import re
import shutil
import struct
import tomllib
from pathlib import Path
from string import Template

import markdown
import nbformat
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "site"
WEB = ROOT / "web"
REPO = "https://github.com/srini2k03/olist-ecommerce-analytics"
REPO_TREE = f"{REPO}/tree/main/"
PORTFOLIO = "https://srini2k03.github.io/"
FONTS = (
    "https://fonts.googleapis.com/css2?family=Geist:wght@300..700"
    "&family=Geist+Mono:wght@400..600&family=Instrument+Serif:ital@0;1&display=swap"
)
NOTEBOOKS = ["01_eda", "02_stats", "03_segmentation", "04_churn"]
TITLES = {
    "01_eda": "01 · Exploratory analysis",
    "02_stats": "02 · Statistics and the simulated A/B test",
    "03_segmentation": "03 · RFM + K-Means segmentation",
    "04_churn": "04 · Repeat-purchase / churn model",
}
BLURBS = {
    "01_eda": "Where Olist makes money and where it loses customers: GMV trend, cohorts, "
    "categories and delivery by state, each chart driven by one of the portable SQL queries.",
    "02_stats": "Part A tests whether late delivery causes bad reviews (bootstrap CIs, Cliff's δ, "
    "ordinal regression). Part B designs and reads a simulated free-shipping A/B test.",
    "03_segmentation": "RFM features, K-Means with k chosen from the elbow, the silhouette and "
    "business sense, clusters named by explicit rules and cross-checked against SQL NTILE RFM.",
    "04_churn": "Who buys again within 180 days: time-based split, point-in-time features, leakage "
    "tests, LightGBM against logistic regression and a recency rule, SHAP reasons.",
}
ASSETS = ("sb-kit.css", "sb-kit.js", "site.css")
PAGE_TEMPLATE = Template((WEB / "page.html").read_text(encoding="utf-8"))
# Pages this script writes (links to them are kept even before they exist).
PAGES = {"index.html", "executive_summary.html", "app", *(f"notebooks/{n}.html" for n in NOTEBOOKS)}

esc = html.escape

ARROW = (
    '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<path d="M7 17 17 7M8 7h9v9"/></svg>'
)
GITHUB = (
    '<svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">'
    '<path d="M12 .5a11.5 11.5 0 0 0-3.64 22.41c.58.1.79-.25.79-.56v-2c-3.2.7-3.88-1.37-3.88-1.37'
    "-.52-1.33-1.28-1.69-1.28-1.69-1.05-.72.08-.7.08-.7 1.16.08 1.77 1.19 1.77 1.19 1.03 1.77 2.7 "
    "1.26 3.36.96.1-.75.4-1.26.73-1.55-2.55-.29-5.24-1.28-5.24-5.69 0-1.26.45-2.29 1.19-3.1-.12-.29"
    "-.52-1.46.11-3.05 0 0 .97-.31 3.17 1.18a11 11 0 0 1 5.77 0c2.2-1.49 3.17-1.18 3.17-1.18.63 "
    "1.59.23 2.76.11 3.05.74.81 1.19 1.84 1.19 3.1 0 4.42-2.69 5.39-5.25 5.68.41.36.78 1.06.78 "
    '2.14v3.17c0 .31.21.67.8.56A11.5 11.5 0 0 0 12 .5Z"/></svg>'
)
THEME_ICONS = (
    '<svg class="icon-sun" width="15" height="15" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true">'
    '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4'
    'M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>'
    '<svg class="icon-moon" width="15" height="15" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
    'aria-hidden="true"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8Z"/></svg>'
)
ANTI_FLASH = (
    "<script>(function(){try{var t=localStorage.getItem('theme');"
    "document.documentElement.dataset.theme=(t==='light'||t==='dark')?t:'dark';}"
    "catch(e){document.documentElement.dataset.theme='dark';}"
    "document.documentElement.classList.remove('no-js');})();</script>"
)


# ---------------------------------------------------------------------------- markdown helpers
def md(text: str) -> str:
    # Python-Markdown nests lists only at 4 spaces; the README and summary indent by 2.
    text = re.sub(r"^ {2}(?=(?:[-*]|\d+\.) )", "    ", text, flags=re.M)
    # ...and needs a blank line between a paragraph and a list that follows it.
    text = re.sub(
        r"^((?![ \t]*(?:[-*]|\d+\.) |[ \t]*$|[|#>]).+)\n(?=(?:[-*]|\d+\.) )",
        r"\1\n\n",
        text,
        flags=re.M,
    )
    return markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])


def inline_md(text: str) -> str:
    return re.sub(r"^<p>(.*)</p>$", r"\1", md(text.strip()), flags=re.S)


def md_sections(text: str, level: int = 2) -> dict[str, str]:
    """Split a markdown document into {heading: body} at the given heading level."""
    out, current, buf = {}, "", []
    mark = "#" * level + " "
    for line in text.splitlines():
        if line.startswith(mark):
            out[current] = "\n".join(buf).strip()
            current, buf = line[len(mark) :].strip(), []
        else:
            buf.append(line)
    out[current] = "\n".join(buf).strip()
    return out


def fix_links(html_: str, depth: int = 0) -> str:
    """Notebook and summary links point at the rendered pages; relative links to files that are
    not published (results/sql, LICENSE, ...) point at the repository instead of a 404."""
    folder = "notebooks/" if depth else ""
    for name in NOTEBOOKS:
        html_ = html_.replace(f'href="notebooks/{name}.ipynb"', f'href="notebooks/{name}.html"')
        html_ = html_.replace(f'href="{name}.ipynb"', f'href="{name}.html"')
    html_ = html_.replace('href="EXECUTIVE_SUMMARY.md"', 'href="executive_summary.html"')

    def fix(m: re.Match) -> str:
        target = m.group(1)
        if not target or re.match(r"[a-z]+:|#", target):
            return m.group(0)
        site_path = posixpath.normpath(folder + target.split("#")[0])
        path = OUT / site_path
        if site_path in PAGES or path.is_file() or (path / "index.html").is_file():
            return m.group(0)
        return f'href="{REPO_TREE}{site_path}"'

    return re.sub(r'href="([^"]*)"', fix, html_)


def png_size(path: Path) -> tuple[int, int] | None:
    with path.open("rb") as f:
        head = f.read(24)
    return struct.unpack(">II", head[16:24]) if head[:8] == b"\x89PNG\r\n\x1a\n" else None


# ---------------------------------------------------------------------------- page chrome
NAV = [
    ("index.html#findings", "Findings", "index"),
    ("executive_summary.html", "Summary", "summary"),
    ("index.html#notebooks", "Notebooks", "notebooks"),
    ("app/", "Dashboard", "app"),
]


def page(title: str, desc: str, body: str, active: str = "", depth: int = 0, head: str = "") -> str:
    up = "../" * depth
    links = "".join(
        f'<a class="sb-nav__link sb-nav__link--hide-sm" href="{up}{href}"'
        f"{' aria-current=page' if key == active else ''}>{label}</a>"
        for href, label, key in NAV
    )
    return PAGE_TEMPLATE.substitute(
        anti_flash=ANTI_FLASH,
        title=esc(title),
        desc=esc(desc),
        fonts=FONTS,
        head=head,
        up=up,
        links=links,
        repo=REPO,
        theme_icons=THEME_ICONS,
        body=body,
        portfolio=PORTFOLIO,
    )


def kicker(idx: str, label: str) -> str:
    return (
        f'<p class="sb-kicker"><span class="sb-kicker__index">{idx}</span>'
        f'<span class="sb-kicker__rule"></span>{label}</p>'
    )


def section(sid: str, idx: str, label: str, title: str, lead: str, body: str) -> str:
    lead_html = f'<p class="sb-lead">{lead}</p>' if lead else ""
    return (
        f'<section class="sb-section" id="{sid}" aria-labelledby="{sid}-h">'
        '<div class="sb-container">'
        f'<div class="section-head sb-reveal">{kicker(idx, esc(label))}'
        f'<h2 class="sb-h2" id="{sid}-h">{title}</h2>{lead_html}</div>{body}</div></section>'
    )


def figure(src: str, path: Path, title: str, sub: str = "", depth: int = 0) -> str:
    size = png_size(path) if path.exists() else None
    dims = f' width="{size[0]}" height="{size[1]}"' if size else ""
    sub_html = f'<span class="sb-card__sub">{sub}</span>' if sub else ""
    href = "../" * depth + src
    return (
        f'<figure class="sb-card fig-card sb-reveal"><figcaption class="card-head">'
        f'<span class="sb-card__title">{esc(title)}</span>{sub_html}</figcaption>'
        f'<a class="fig-plate" href="{href}" target="_blank" rel="noopener" '
        'title="Open the full-size image">'
        f'<img src="{href}" alt="{esc(title)}" loading="lazy" decoding="async"{dims}></a></figure>'
    )


# ---------------------------------------------------------------------------- landing page
def headline_stats() -> list[tuple[str, str, str]]:
    """Four stat tiles, read from the committed results (same rounding as the README)."""
    res = ROOT / "results"
    stats = json.loads((res / "metrics" / "stats.json").read_text())["late_vs_ontime"]
    segs = json.loads((res / "metrics" / "segments.json").read_text())["segments"]
    churn = json.loads((ROOT / "app" / "data" / "model_card.json").read_text())["test_metrics"]
    with (res / "sql" / "13_late_delivery_hotspots.csv").open() as f:
        lanes = list(csv.DictReader(f))
    top_lane = max(lanes, key=lambda r: float(r["share_of_all_late"]))
    big = next(s for s in segs if s["segment"] == "At-risk big spenders")
    delta = f"{stats['cliffs_delta']:.2f}".replace("-", "−")
    return [
        (
            f"{stats['mean_a']:.2f}★",
            "Average review when late",
            f"vs {stats['mean_b']:.2f}★ on time · Cliff's δ = {delta}",
        ),
        (
            f"{float(top_lane['share_of_all_late']):.1%}",
            "Of all late deliveries on one lane",
            f"{esc(top_lane['lane']).replace('-&gt;', '→')} · SQL 13",
        ),
        (
            f"{big['revenue_share']:.1%}",
            f"Of revenue from {big['customer_share']:.1%} of customers",
            "“At-risk big spenders” segment",
        ),
        (
            f"{churn['capture_top10']:.1%}",
            "Of repeat buyers in the model's top 10%",
            f"{churn['lift_top10']:.2f}x lift · ROC-AUC {churn['roc_auc']:.3f}, a weak signal",
        ),
    ]


def findings(readme: dict[str, str]) -> list[dict]:
    rows = []
    for line in readme.get("Key findings", "").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 3 and cells[0].isdigit():
            m = re.match(r"\*\*(.+?)\*\*\s*(.*)", cells[1], re.S)
            title, body = (m.group(1), m.group(2)) if m else ("", cells[1])
            if title.endswith(":"):  # "**Simulated A/B test, ...:** free shipping ..."
                title, body = title[:-1] + ".", body[:1].upper() + body[1:]
            rows.append({"n": int(cells[0]), "title": title, "body": body, "evidence": cells[2]})
    return rows


def index_page(readme_text: str) -> str:
    readme = md_sections(readme_text)
    tiles = "".join(
        f'<div class="sb-stat" role="listitem"><div class="sb-stat__value">{v}</div>'
        f'<div class="sb-stat__label">{lab}</div><div class="sb-stat__ctx">{ctx}</div></div>'
        for v, lab, ctx in headline_stats()
    )
    cards = []
    for f in findings(readme):
        simulated = "simulated" in (f["title"] + f["evidence"]).lower()
        flag = '<span class="sb-pill sb-pill--pending">Simulated</span>' if simulated else ""
        cards.append(
            f'<article class="sb-card sb-spotlight finding sb-reveal">'
            f'<header class="finding__top"><span class="finding__n">{f["n"]:02d}</span>'
            f"{flag}</header>"
            f'<h3 class="finding__title">{inline_md(f["title"])}</h3>'
            f'<p class="finding__body">{inline_md(f["body"])}</p>'
            '<p class="finding__ev"><span class="sb-label">Evidence</span> '
            f"{inline_md(f['evidence'])}</p>"
            f"</article>"
        )
    figs = re.findall(
        r'<img src="(results/figures/[^"]+)"[^>]*alt="([^"]+)"', readme.get("Key findings", "")
    )
    fig_html = "".join(figure(src, ROOT / src, alt) for src, alt in figs)
    nb_cards = "".join(
        f'<a class="sb-card sb-spotlight nb-card sb-reveal" href="notebooks/{n}.html">'
        f'<span class="nb-card__n">{TITLES[n].split(" · ")[0]}</span>'
        f'<span class="nb-card__title">{esc(TITLES[n].split(" · ", 1)[1])}</span>'
        f'<span class="nb-card__text">{esc(BLURBS[n])}</span>'
        f'<span class="nb-card__go">Read the executed notebook →</span></a>'
        for n in NOTEBOOKS
    )
    sql = md_sections(readme.get("SQL library (`sql/`)", ""), 3)
    sql_html = md(sql.get("", ""))
    methods = md(readme.get("Methods in brief", "").split("### Being candid")[0])
    candid = md_sections(readme.get("Methods in brief", ""), 3).get(
        "Being candid about the churn model", ""
    )
    honesty = md(readme.get("Honesty notes", ""))
    shot = ROOT / "results" / "figures" / "app_churn.png"
    app_fig = (
        figure("results/figures/app_churn.png", shot, "Churn-risk lookup in the dashboard")
        if shot.exists()
        else ""
    )
    findings_html = section(
        "findings",
        "01",
        "Key findings",
        'Seven findings, each with its <span class="sb-serif">evidence</span>.',
        "Every number below comes from executed code on the real data: SQL snapshots in "
        "<code>results/sql/</code> and the executed notebooks.",
        f'<div class="finding-grid">{"".join(cards)}</div><div class="fig-grid">{fig_html}</div>',
    )
    extra_cards = (
        '<div class="sb-grid-2 duo">'
        '<a class="sb-card sb-spotlight nb-card sb-reveal" href="executive_summary.html">'
        '<span class="nb-card__n">One page</span>'
        '<span class="nb-card__title">Executive summary</span>'
        '<span class="nb-card__text">The bottom line and five recommendations for the business, '
        'with the caveats.</span><span class="nb-card__go">Read the summary →</span></a>'
        '<a class="sb-card sb-spotlight nb-card sb-reveal" href="app/">'
        '<span class="nb-card__n">Live</span>'
        '<span class="nb-card__title">Streamlit dashboard</span>'
        '<span class="nb-card__text">KPIs, revenue trend, cohort retention, segments, delivery '
        "impact and a churn-risk lookup by customer, running entirely in your browser.</span>"
        '<span class="nb-card__go">Open the dashboard →</span></a></div>'
    )
    notebooks_html = section(
        "notebooks",
        "02",
        "Notebooks",
        'Four notebooks, executed on the <span class="sb-serif">real</span> data.',
        "Rendered from the committed, executed notebooks. CI re-executes them on every push.",
        f'<div class="nb-grid">{nb_cards}</div>{extra_cards}<div class="stack">{app_fig}</div>',
    )
    methods_html = section(
        "methods",
        "03",
        "Methods",
        'One SQL text, two <span class="sb-serif">engines</span>.',
        "",
        f'<div class="split"><div class="prose sb-reveal">{methods}</div>'
        '<div class="stack stack--tight">'
        '<div class="sb-callout sb-callout--info sb-reveal" role="note"><div>'
        f'<p class="sb-label">Being candid about the churn model</p>{md(candid)}</div></div>'
        '<div class="sb-callout sb-callout--warn sb-reveal" role="note"><div>'
        f'<p class="sb-label">Honesty notes</p>{honesty}</div></div></div></div>'
        '<article class="sb-card sb-reveal sql-card"><header class="card-head">'
        '<h3 class="sb-card__title">SQL library</h3>'
        '<p class="sb-card__sub">16 documented queries in '
        "<code>sql/</code>, each with its business question, method, expected output and dialect "
        f'notes.</p></header><div class="prose prose--table">{sql_html}</div>'
        f'<p class="sb-card__foot">Reproduce, data licence and deployment notes: the '
        f'<a href="{REPO}#readme">README on GitHub</a>.</p></article>',
    )
    body = f"""
<section class="sb-hero" aria-labelledby="top-h">
  <div class="sb-container">
    {kicker("Analytics · Statistics", "2026 · Olist marketplace, 99,441 orders")}
    <h1 class="sb-h1" id="top-h">Why Olist's customers don't come
      <span class="sb-serif">back</span>.</h1>
    <p class="sb-lead">Portable SQL, causal-minded statistics, segmentation and a churn model on
      the real Olist Brazilian e-commerce data: 99,441 orders across 9 tables, from Sept 2016 to
      Oct 2018. It ends with a dashboard, a one-page summary and a recommendation not to launch
      free shipping as designed.</p>
    <div class="sb-hero__meta">
      <span class="sb-pill sb-pill--live">Dashboard live (in-browser)</span>
      <span class="sb-pill">Results committed · real data</span>
      <span class="sb-pill sb-pill--pending">A/B test simulated</span>
    </div>
    <div class="sb-hero__actions">
      <a class="sb-btn sb-btn--primary" href="app/">Open the dashboard {ARROW}</a>
      <a class="sb-btn" href="executive_summary.html">Executive summary</a>
      <a class="sb-btn sb-btn--ghost" href="{REPO}">{GITHUB} Code</a>
    </div>
    <p class="hero-note">The dashboard is Streamlit compiled to WebAssembly: the first load takes
      10–30 s while Python starts in your browser.</p>
    <div class="sb-stats hero-stats sb-reveal" role="list">{tiles}</div>
  </div>
</section>
{findings_html}
{notebooks_html}
{methods_html}
"""
    body = re.sub(
        r"<table>", '<div class="sb-table-wrap" tabindex="0"><table class="sb-table">', body
    )
    body = body.replace("</table>", "</table></div>")
    return page(
        "Olist e-commerce analytics",
        "SQL, statistics, segmentation and a churn model on 99,441 real Olist orders, with key "
        "findings, an executive summary, the executed notebooks and an in-browser dashboard.",
        fix_links(body),
        active="index",
    )


# ---------------------------------------------------------------------------- executive summary
def rec_body(text: str) -> str:
    """A recommendation's bullets. Each "**Label:** text" shows its label on a line of its own,
    so the colon goes and the text after it starts with a capital letter."""
    out = md(text)
    out = re.sub(r"(<li><strong>[^<]+?):(</strong>)", r"\1\2", out)
    return re.sub(r"(<li><strong>[^<]+</strong>\s*)([a-z])", lambda m: m[1] + m[2].upper(), out)


def summary_page(text: str) -> str:
    title_line = next(
        (ln for ln in text.splitlines() if ln.startswith("# ")), "# Executive summary"
    )
    subtitle = title_line[2:].split(":", 1)[1].strip() if ":" in title_line else ""
    secs = md_sections(text)
    intro_paras = [p.strip() for p in secs.get("", "").split("\n\n") if p.strip()]
    intro_paras = [p for p in intro_paras if not p.startswith("# ")]
    bottom = next((p for p in intro_paras if p.startswith("**Bottom line")), "")
    rest = [p for p in intro_paras if p is not bottom]
    bottom_html = inline_md(re.sub(r"^\*\*Bottom line\.\*\*\s*", "", bottom))

    recs_md = secs.get("Recommendations", "")
    blocks = re.split(r"^\*\*(\d+)\.\s+(.+?)\*\*\s*$", recs_md, flags=re.M)
    recs = []
    for i in range(1, len(blocks) - 2, 3):
        n, title, rest_md = blocks[i], blocks[i + 1], blocks[i + 2]
        sim = "simulated" in rest_md.lower()
        flag = '<span class="sb-pill sb-pill--pending">Simulated input</span>' if sim else ""
        recs.append(
            f'<article class="sb-card rec sb-reveal" id="rec-{n}">'
            f'<div class="rec__n" aria-hidden="true">{int(n):02d}</div>'
            '<div class="rec__main"><header class="rec__head"><h3 class="rec__title">'
            f'<span class="sr-only">Recommendation {n}: </span>{inline_md(title)}</h3>'
            f"{flag}</header>"
            f'<div class="rec__body">{rec_body(rest_md)}</div></div></article>'
        )
    caveats = secs.get("Caveats", "")
    details = ""
    m = re.search(r"^Details:.*$", caveats, flags=re.M)
    if m:
        details, caveats = m.group(0), caveats.replace(m.group(0), "").strip()
    if not recs:  # unexpected structure: fall back to the plain rendering
        recs = [f'<div class="prose sb-reveal">{md(recs_md)}</div>']
    body = f"""
<section class="sb-hero sb-hero--page" aria-labelledby="top-h">
  <div class="sb-container sb-container--narrow">
    {kicker("Executive summary", esc(subtitle) or "Olist marketplace")}
    <h1 class="sb-h1 sb-h1--page" id="top-h">Keep the customers you already
      <span class="sb-serif">win</span>.</h1>
    <div class="bottom-line sb-reveal"><p class="sb-label">Bottom line</p><p>{bottom_html}</p></div>
    {"".join(f'<p class="scope">{inline_md(p)}</p>' for p in rest)}
  </div>
</section>
<section class="sb-section sb-section--tight" aria-labelledby="recs-h">
  <div class="sb-container sb-container--narrow">
    <div class="section-head sb-reveal">{kicker("01", "Recommendations")}
      <h2 class="sb-h2" id="recs-h">Five <span class="sb-serif">recommendations</span>.</h2></div>
    <div class="recs">{"".join(recs)}</div>
    <div class="sb-callout sb-callout--warn caveats sb-reveal" role="note"><div>
      <p class="sb-label">Caveats</p>{md(caveats)}</div></div>
    {f'<p class="details sb-reveal">{inline_md(details)}</p>' if details else ""}
  </div>
</section>"""
    return page(
        "Executive summary · Olist analytics",
        "One-page executive summary: the bottom line and five recommendations from the Olist "
        "marketplace analysis.",
        fix_links(body),
        active="summary",
    )


# ---------------------------------------------------------------------------- notebooks
def export_notebook(name: str) -> tuple[str, list[str]]:
    """nbconvert's lab export, split into its <main> content and its stylesheets. Its scripts
    (require.js, MathJax, mermaid) are dropped: the notebooks use no widgets, LaTeX or diagrams,
    and MathJax would read currency such as "R$ 9.60 ... R$ 1.44" as inline math."""
    nb = nbformat.read(ROOT / "notebooks" / f"{name}.ipynb", as_version=4)
    full, _ = HTMLExporter(template_name="lab").from_notebook_node(
        nb, resources={"metadata": {"name": TITLES[name]}}
    )
    head = full[: full.index("<body")]
    styles = re.findall(r"<style[^>]*>(.*?)</style>", head, re.S)
    main = re.search(r"<main>(.*)</main>", full, re.S)
    content = main.group(1) if main else full[full.index(">", full.index("<body")) + 1 :]
    content = content.replace(
        'alt="No description has been provided for this image"',
        'alt="Chart output of the cell above"',
    )
    content = re.sub(r"<table\b", '<div class="nb-table"><table', content)
    content = content.replace("</table>", "</table></div>")
    return content, styles


def notebook_page(name: str, content: str) -> str:
    i = NOTEBOOKS.index(name)
    prev_nb = NOTEBOOKS[i - 1] if i else None
    next_nb = NOTEBOOKS[i + 1] if i + 1 < len(NOTEBOOKS) else None

    def pager(n: str | None, label: str, cls: str) -> str:
        if not n:
            return f'<span class="pager__item pager__item--empty {cls}"></span>'
        return (
            f'<a class="sb-card pager__item {cls}" href="{n}.html">'
            f'<span class="sb-label">{label}</span>'
            f'<span class="pager__title">{esc(TITLES[n])}</span></a>'
        )

    num = TITLES[name].split(" · ")[0]
    prev_html = pager(prev_nb, "← Previous", "pager__item--prev")
    next_html = pager(next_nb, "Next →", "pager__item--next")
    body = f"""
<div class="sb-container nb-shell">
  <div class="nb-bar">
    {kicker(f"Notebook {num} / {len(NOTEBOOKS):02d}", "Executed on the real data")}
    <p class="nb-bar__links">
      <a class="sb-btn sb-btn--sm" href="{REPO}/blob/main/notebooks/{name}.ipynb">{GITHUB}
        View .ipynb</a>
      <a class="sb-btn sb-btn--sm sb-btn--ghost" href="../index.html#notebooks">All notebooks</a>
    </p>
  </div>
  <article class="nb jp-Notebook" aria-label="{esc(TITLES[name])}">{content}</article>
  <nav class="pager" aria-label="Notebooks">{prev_html}{next_html}</nav>
</div>"""
    return page(
        f"{TITLES[name]} · Olist analytics",
        f"Executed notebook: {TITLES[name]}.",
        fix_links(body, depth=1),
        active="notebooks",
        depth=1,
        head='<link rel="stylesheet" href="nbconvert.css">\n',
    )


# ---------------------------------------------------------------------------- dashboard (stlite)
# The Streamlit dashboard also runs fully in the browser on GitHub Pages via stlite
# (Streamlit on Pyodide), so it works without a server or a Streamlit Cloud account.
STLITE_VERSION = "1.10.0"
APP_TEMPLATE = Template((WEB / "app.html").read_text(encoding="utf-8"))


FONT_KEYS = ("font", "headingFont", "codeFont")


def streamlit_theme() -> dict:
    """The [theme] table of .streamlit/config.toml (also used by `streamlit run` and Streamlit
    Cloud), flattened to stlite's streamlitConfig keys.

    Font values of the form ``Family:https://...`` make Streamlit's frontend inject the font
    stylesheet through react-helmet, which crashes stlite 1.10.0 before the app renders
    ("Cannot read properties of undefined (reading 'add')" in HelmetDispatcher). The stlite
    page (web/app.html) loads the same Google Fonts itself, so only the family name is passed.
    """
    cfg = ROOT / ".streamlit" / "config.toml"
    theme = tomllib.loads(cfg.read_text()).get("theme", {}) if cfg.exists() else {}
    out = {}
    for key, value in theme.items():
        if isinstance(value, dict | list):
            continue
        if key in FONT_KEYS and isinstance(value, str) and ":http" in value:
            value = value.split(":http", 1)[0].strip("'\"")
        out[f"theme.{key}"] = value
    return out


def build_browser_app() -> None:
    """Copy the dashboard and its prebuilt data into site/app and add a stlite loader."""
    app_out = OUT / "app"
    (app_out / "data").mkdir(parents=True)
    shutil.copy2(ROOT / "app" / "streamlit_app.py", app_out / "streamlit_app.py")
    files = ["streamlit_app.py"]
    for f in sorted((ROOT / "app" / "data").iterdir()):
        if f.is_file():
            shutil.copy2(f, app_out / "data" / f.name)
            files.append(f"data/{f.name}")
    (app_out / "index.html").write_text(
        APP_TEMPLATE.substitute(
            version=STLITE_VERSION,
            files=json.dumps(files),
            config=json.dumps(streamlit_theme()),
            fonts=FONTS,
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------- main
def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "notebooks").mkdir(parents=True)
    shutil.copytree(ROOT / "results" / "figures", OUT / "results" / "figures")
    for name in ASSETS:
        shutil.copy2(WEB / name, OUT / name)
    build_browser_app()

    nb_css = None
    for name in NOTEBOOKS:
        content, styles = export_notebook(name)
        nb_css = nb_css or "\n".join(styles)
        (OUT / "notebooks" / f"{name}.html").write_text(
            notebook_page(name, content), encoding="utf-8"
        )
    (OUT / "notebooks" / "nbconvert.css").write_text(
        "/* JupyterLab styles from nbconvert's lab template; themed by ../site.css */\n" + nb_css,
        encoding="utf-8",
    )
    (OUT / "executive_summary.html").write_text(
        summary_page((ROOT / "EXECUTIVE_SUMMARY.md").read_text(encoding="utf-8")), encoding="utf-8"
    )
    (OUT / "index.html").write_text(
        index_page((ROOT / "README.md").read_text(encoding="utf-8")), encoding="utf-8"
    )
    (OUT / ".nojekyll").write_text("")
    print(f"site built in {OUT}: {sorted(p.name for p in OUT.iterdir())}")


if __name__ == "__main__":
    main()

"""Theme-aware styling and small UI building blocks shared by the pages.

Colours are the portfolio's tokens (web/sb-kit.css) for the active Streamlit theme, so the
dashboard follows the site in dark (default) and light mode.
"""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from olist.metrics import format_value, metric_specs

DARK = {
    "bg": "#07090c",
    "elev": "#0d1117",
    "border": "#232831",
    "text": "#eef1f5",
    "muted": "#8b95a3",
    "accent": "#5eeaa8",
    "project": "#ffb547",
    "series1": "#1ea372",
    "series2": "#d96a33",
    "series_muted": "#3a414c",
    "grid": "#1c222b",
    "danger": "#ff6b6b",
}
LIGHT = {
    "bg": "#f6f6f2",
    "elev": "#ffffff",
    "border": "#e3e4e0",
    "text": "#0c0f14",
    "muted": "#5b6470",
    "accent": "#0a7d52",
    "project": "#a55f00",
    "series1": "#0f9e6a",
    "series2": "#e0662f",
    "series_muted": "#c9ccd1",
    "grid": "#e6e7ea",
    "danger": "#c23a3a",
}
BADGES = {
    "simulated": ("Simulated", "Not a real experiment: effects are assumptions."),
    "hypothetical": ("Hypothetical", "What would follow if the labelled assumptions held."),
    "estimate": ("Estimate", "Built from labelled assumptions, not observed in the data."),
    "full": ("Full data", "Computed on the whole dataset; the sidebar filters do not apply."),
    "observed": ("Observed", "Computed from the filtered orders."),
}


def palette() -> dict:
    try:
        kind = st.context.theme.type
    except AttributeError:  # older Streamlit
        kind = "dark"
    return LIGHT if kind == "light" else DARK


def inject_css() -> None:
    p = palette()
    st.html(
        f"""<style>
h1, h2, h3 {{ letter-spacing: -0.03em; }}
h1 {{ font-weight: 600 !important; }}
[data-testid="stMetric"] {{ background: {p["elev"]}; border: 1px solid {p["border"]};
  border-radius: 16px; padding: 0.85rem 1.05rem; height: 100%; }}
[data-testid="stMetricLabel"] p {{ font-family: "Geist Mono", ui-monospace, monospace;
  font-size: 10.5px; letter-spacing: 0.12em; text-transform: uppercase; color: {p["muted"]};
  white-space: normal; overflow: visible; text-overflow: clip; line-height: 1.35; }}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] > div {{ overflow: visible; }}
[data-testid="stMetricValue"] {{ font-variant-numeric: tabular-nums; letter-spacing: -0.03em;
  font-size: 1.6rem; }}
[data-testid="stMetricValue"] > div {{ overflow: visible; text-overflow: clip; }}
[data-testid="stMarkdownContainer"] p.sb-kicker {{ display: flex; flex-wrap: wrap;
  align-items: center; gap: 0.75rem; margin: 0; font: 500 11px/1.4 "Geist Mono", ui-monospace,
  monospace; letter-spacing: 0.2em; text-transform: uppercase; color: {p["muted"]}; }}
.sb-kicker .k-index {{ color: {p["project"]}; }}
.sb-kicker .k-rule {{ width: 2rem; height: 1px; background: {p["border"]}; }}
.sb-badge {{ display: inline-block; padding: 0.1rem 0.55rem; border-radius: 999px;
  font: 500 10.5px/1.6 "Geist Mono", ui-monospace, monospace; letter-spacing: 0.08em;
  text-transform: uppercase; border: 1px solid {p["border"]}; color: {p["muted"]};
  margin-right: 0.4rem; }}
.sb-badge.simulated, .sb-badge.hypothetical, .sb-badge.estimate {{ color: {p["project"]};
  border-color: {p["project"]}; }}
.sb-badge.full {{ color: {p["accent"]}; border-color: {p["accent"]}; }}
.sb-note {{ color: {p["muted"]}; font-size: 0.86rem; margin: 0.15rem 0 0.6rem; }}
.sb-dq {{ font: 500 11px/1.5 "Geist Mono", ui-monospace, monospace; color: {p["muted"]}; }}
.sb-dq b {{ color: {p["accent"]}; }}
</style>"""
    )


def esc(text: str) -> str:
    """Escape '$' so Streamlit markdown does not read "R$ 1 ... R$ 2" as LaTeX math."""
    return text.replace("$", "\\$")


def header(index: str, label: str, title: str, lead: str = "") -> None:
    st.markdown(
        f'<p class="sb-kicker"><span class="k-index">{index}</span><span class="k-rule"></span>'
        f"{label}</p>",
        unsafe_allow_html=True,
    )
    st.title(title)
    if lead:
        st.caption(lead)


def badge(kind: str, text: str = "") -> None:
    label, default = BADGES[kind]
    st.markdown(
        f'<p class="sb-note"><span class="sb-badge {kind}">{label}</span>{text or default}</p>',
        unsafe_allow_html=True,
    )


def kpis(values: dict[str, float], names: list[str], spec: dict, per_row: int | None = None):
    """One st.metric per metric name; the tooltip is the metric definition from the spec.
    Rows are balanced: 6 cards -> 2 x 3, 8 -> 2 x 4, 12 -> 3 x 4."""
    specs = metric_specs(spec)
    if per_row is None:
        per_row = 4 if len(names) % 4 == 0 else 3 if len(names) % 3 == 0 else 4
    for start in range(0, len(names), per_row):
        row = names[start : start + per_row]
        cols = st.columns(len(row))
        for col, name in zip(cols, row, strict=True):
            s = specs[name]
            help_text = s.definition + (f" Caveat: {s.caveats}" if s.caveats else "")
            label = s.label if s.kind == "observed" else s.label.replace(" (estimate)", "*")
            col.metric(label, format_value(values.get(name, math.nan), s.unit), help=help_text)


def download(df: pd.DataFrame, name: str, label: str = "Download CSV") -> None:
    st.download_button(
        label,
        df.to_csv(index=False).encode("utf-8"),
        file_name=f"olist_{name}.csv",
        mime="text/csv",
        key=f"dl_{name}",
        icon=":material/download:",
    )


def empty(orders: pd.DataFrame, min_rows: int = 1) -> bool:
    """Show an empty state and return True when the selection has too few orders."""
    if len(orders) < min_rows:
        st.info(
            "No orders match the current filters. Widen the date range or clear some filters "
            "in the sidebar.",
            icon=":material/filter_alt_off:",
        )
        return True
    return False


def table(df: pd.DataFrame, formats: dict[str, str] | None = None, height: int | None = None):
    """Dataframe with simple per-column number formats ("R$", "%", "int", "1", "2")."""
    cfg = {}
    for col, f in (formats or {}).items():
        if f == "R$":
            cfg[col] = st.column_config.NumberColumn(col, format="R$ %.2f")
        elif f == "%":
            cfg[col] = st.column_config.NumberColumn(col, format="percent")
        elif f == "int":
            cfg[col] = st.column_config.NumberColumn(col, format="localized")
        elif f in ("1", "2", "3"):
            cfg[col] = st.column_config.NumberColumn(col, format=f"%.{f}f")
    kwargs = {"height": height} if height else {}
    st.dataframe(df, hide_index=True, width="stretch", column_config=cfg, **kwargs)

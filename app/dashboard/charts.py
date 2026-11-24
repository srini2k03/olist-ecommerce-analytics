"""Altair charts in the portfolio's chart tokens (both themes)."""

from __future__ import annotations

import altair as alt
import pandas as pd

from dashboard.ui import palette


def _cfg(chart: alt.Chart, height: int) -> alt.Chart:
    p = palette()
    return (
        chart.properties(height=height)
        .configure(background="transparent", font="Geist")
        .configure_view(stroke=None)
        .configure_axis(
            gridColor=p["grid"],
            domainColor=p["grid"],
            tickColor=p["grid"],
            labelColor=p["muted"],
            titleColor=p["muted"],
            labelFontSize=11,
            titleFontSize=11,
            titleFontWeight=500,
        )
        .configure_legend(labelColor=p["muted"], titleColor=p["muted"], orient="top")
    )


def line(
    df: pd.DataFrame,
    x: str,
    y: str,
    title_y: str,
    fmt: str = ",.0f",
    color: str | None = None,
    height: int = 280,
    tooltip: list | None = None,
    y_format: str | None = None,
) -> alt.Chart:
    p = palette()
    enc = {
        "x": alt.X(f"{x}:T", title=None),
        "y": alt.Y(f"{y}:Q", title=title_y, axis=alt.Axis(format=y_format or "~s")),
        "tooltip": tooltip or [alt.Tooltip(f"{x}:T", format="%b %Y"), alt.Tooltip(y, format=fmt)],
    }
    if color:
        enc["color"] = alt.Color(
            f"{color}:N",
            title=None,
            scale=alt.Scale(range=[p["series1"], p["series2"], p["project"], p["series_muted"]]),
        )
        mark = alt.Chart(df).mark_line(point=True, strokeWidth=2)
    else:
        mark = alt.Chart(df).mark_line(
            color=p["series1"], strokeWidth=2, point=alt.OverlayMarkDef(color=p["series1"], size=36)
        )
    return _cfg(mark.encode(**enc), height)


def bars(
    df: pd.DataFrame,
    x: str,
    y: str,
    title_x: str = "",
    fmt: str = ",.0f",
    horizontal: bool = True,
    height: int = 320,
    color: str | None = None,
    tooltip: list | None = None,
    x_format: str = "~s",
    sort: str | list = "-x",
    error: tuple[str, str] | None = None,
) -> alt.Chart:
    """Horizontal bars by default: y is the category, x the value."""
    p = palette()
    value, cat = (x, y) if horizontal else (y, x)
    base = alt.Chart(df)
    enc_val = alt.X(f"{value}:Q", title=title_x, axis=alt.Axis(format=x_format))
    enc_cat = alt.Y(f"{cat}:N", title=None, sort=sort)
    if not horizontal:
        enc_val = alt.Y(f"{value}:Q", title=title_x, axis=alt.Axis(format=x_format))
        enc_cat = alt.X(f"{cat}:N", title=None, sort=sort)
    enc = {"x": enc_val, "y": enc_cat} if horizontal else {"x": enc_cat, "y": enc_val}
    enc["tooltip"] = tooltip or [cat, alt.Tooltip(value, format=fmt)]
    if color:
        enc["color"] = alt.Color(
            f"{color}:N",
            title=None,
            scale=alt.Scale(range=[p["series1"], p["series2"], p["project"], p["series_muted"]]),
        )
        if horizontal:
            enc["yOffset"] = f"{color}:N"
        layer = base.mark_bar(cornerRadiusEnd=3).encode(**enc)
    else:
        layer = base.mark_bar(cornerRadiusEnd=3, color=p["series1"]).encode(**enc)
    if error:
        lo, hi = error
        rule = base.mark_rule(color=p["text"], opacity=0.6).encode(
            x=alt.X(f"{lo}:Q"), x2=f"{hi}:Q", y=alt.Y(f"{cat}:N", sort=sort)
        )
        layer = layer + rule
    return _cfg(layer, height)


def heatmap(
    df: pd.DataFrame,
    x: str,
    y: str,
    value: str,
    title: str,
    fmt: str = ".2%",
    height: int = 460,
    domain: list | None = None,
    tooltip: list | None = None,
) -> alt.Chart:
    p = palette()
    scale = alt.Scale(range=[p["grid"], p["series1"], p["accent"]])
    if domain is not None:
        scale = alt.Scale(range=[p["grid"], p["series1"], p["accent"]], domain=domain)
    chart = (
        alt.Chart(df)
        .mark_rect(stroke=p["bg"], strokeWidth=1)
        .encode(
            x=alt.X(f"{x}:O", title="Months since first order"),
            y=alt.Y(f"{y}:O", title=None),
            color=alt.Color(f"{value}:Q", title=title, scale=scale, legend=alt.Legend(format=fmt)),
            tooltip=tooltip or [y, x, alt.Tooltip(value, format=fmt)],
        )
    )
    return _cfg(chart, height)


def scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    title_x: str,
    title_y: str,
    color: str | None = None,
    highlight: str | None = None,
    height: int = 320,
    log_y: bool = False,
    tooltip: list | None = None,
    size: int = 26,
) -> alt.Chart:
    p = palette()
    enc = {
        "x": alt.X(f"{x}:Q", title=title_x),
        "y": alt.Y(
            f"{y}:Q", title=title_y, scale=alt.Scale(type="log") if log_y else alt.Undefined
        ),
        "tooltip": tooltip or [x, y],
    }
    if highlight:
        enc["color"] = alt.condition(
            f"datum.{highlight}", alt.value(p["series2"]), alt.value(p["series_muted"])
        )
        enc["opacity"] = alt.condition(f"datum.{highlight}", alt.value(0.9), alt.value(0.4))
    elif color:
        enc["color"] = alt.Color(f"{color}:N", title=None)
    mark = alt.Chart(df).mark_circle(size=size, color=p["series1"])
    return _cfg(mark.encode(**enc), height)


def funnel(
    df: pd.DataFrame,
    band_df: pd.DataFrame,
    n: str,
    rate: str,
    lo: str,
    hi: str,
    overall: float,
    height: int = 340,
    tooltip: list | None = None,
) -> alt.Chart:
    """Rate vs volume with the Wilson band around the overall rate (``band_df``: a grid of
    volumes): small sellers scatter widely by chance alone."""
    p = palette()
    xs = alt.Scale(type="log")
    pts = (
        alt.Chart(df)
        .mark_circle(size=28, color=p["series1"], opacity=0.75)
        .encode(
            x=alt.X(f"{n}:Q", title="Delivered orders (log)", scale=xs, axis=alt.Axis(tickCount=6)),
            y=alt.Y(f"{rate}:Q", title="Late rate", axis=alt.Axis(format="%")),
            tooltip=tooltip or [n, alt.Tooltip(rate, format=".1%")],
        )
    )
    band = (
        alt.Chart(band_df)
        .mark_area(opacity=0.22, color=p["series2"])
        .encode(x=alt.X(f"{n}:Q", scale=xs), y=f"{lo}:Q", y2=f"{hi}:Q")
    )
    rule = (
        alt.Chart(pd.DataFrame({"y": [overall]}))
        .mark_rule(color=p["series2"], strokeDash=[4, 4])
        .encode(y="y:Q")
    )
    return _cfg(band + rule + pts, height)

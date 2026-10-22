"""Shared look for the benchmark dashboards: a light theme, one Plotly template and fixed colours per model and
device (a colour follows its entity, so filtering never repaints the remaining series). The categorical slots are a
colour-blind-checked palette.
"""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ---------------------------------------------------------------------------------------------------------- colours
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE, PAGE, BORDER = "#e1e0d9", "#c3c2b7", "#fcfcfb", "#f9f9f7", "rgba(11,11,11,0.10)"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE, ORANGE, AQUA, YELLOW, VIOLET = SERIES[0], SERIES[1], SERIES[2], SERIES[3], SERIES[6]
FULL = "#6f6e69"                     # full cache: the neutral reference, not a categorical slot
EVICT, RKV, HYBRID = ORANGE, BLUE, AQUA
# sequential (one hue, light -> dark) and diverging (red <- grey -> blue) scales
SEQ_BLUE = ["#eef4fc", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = [[0.0, "#a52b2a"], [0.2, "#e34948"], [0.4, "#f3b2ac"], [0.5, "#f0efec"], [0.6, "#b7d3f6"],
             [0.8, "#3987e5"], [1.0, "#184f95"]]
FONT = 'Inter, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'

pio.templates["bench"] = go.layout.Template(
    layout=dict(
        font=dict(family=FONT, size=13, color=INK2),
        title=dict(font=dict(size=15, color=INK), x=0, xanchor="left", y=0.98, yanchor="top", pad=dict(b=8)),
        colorway=SERIES, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff",
        xaxis=dict(showgrid=False, linecolor=AXIS, ticks="outside", tickcolor=AXIS, ticklen=4, zeroline=False,
                   title=dict(font=dict(color=INK2, size=12)), automargin=True),
        yaxis=dict(gridcolor=GRID, gridwidth=1, zeroline=False, linecolor="rgba(0,0,0,0)", ticks="",
                   title=dict(font=dict(color=INK2, size=12)), automargin=True),
        legend=dict(orientation="h", yref="container", yanchor="bottom", y=0.0, xanchor="left", x=0,
                    title=dict(text=""), tracegroupgap=4,
                    font=dict(size=12, color=INK2), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=AXIS, font=dict(family=FONT, size=12, color=INK)),
        hovermode="closest", barcornerradius=4, bargap=0.22, bargroupgap=0.06,
        margin=dict(t=64, l=56, r=16, b=104),
        coloraxis=dict(colorbar=dict(outlinewidth=0, thickness=12, tickfont=dict(color=INK2))),
        annotationdefaults=dict(font=dict(color=INK2, size=12)),
        shapedefaults=dict(line=dict(color=FULL, width=1.5)),
    ),
    data=dict(
        scatter=[go.Scatter(line=dict(width=2), marker=dict(size=8, line=dict(width=1.5, color=SURFACE)))],
        bar=[go.Bar(marker=dict(line=dict(width=0)), textfont=dict(color=INK2, size=11), cliponaxis=False)],
        heatmap=[go.Heatmap(xgap=2, ygap=2)],
    ),
)
pio.templates.default = "plotly_white+bench"

CSS = f"""
<style>
.block-container {{padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1480px;}}
h1, h2, h3 {{letter-spacing: -0.015em; color: {INK};}}
h1 {{font-weight: 680 !important;}}
h3 {{font-weight: 620 !important; margin-top: 0.6rem;}}
[data-testid="stCaptionContainer"], .stCaption {{color: {INK2};}}
[data-testid="stMetric"] {{background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 12px;
    padding: 14px 18px 12px 18px;}}
[data-testid="stMetricLabel"] p {{color: {INK2}; font-size: 0.82rem; font-weight: 500;}}
[data-testid="stMetricValue"] {{font-size: 1.7rem; font-weight: 650; color: {INK};}}
.stTabs [role="tablist"] {{gap: 4px; border-bottom: 1px solid {GRID};}}
.stTabs [role="tab"] {{padding: 8px 14px; font-weight: 520; color: {INK2};}}
.stTabs [aria-selected="true"] {{color: {INK};}}
[data-testid="stSidebar"] {{border-right: 1px solid {GRID};}}
[data-testid="stExpander"] details {{border-radius: 10px; border-color: {GRID};}}
.bench-hero {{padding: 4px 0 14px 0; border-bottom: 1px solid {GRID}; margin-bottom: 14px;}}
.bench-hero .eyebrow {{font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; color: {BLUE};
    font-weight: 650;}}
.bench-hero h1 {{margin: 2px 0 6px 0; padding: 0; font-size: 2.05rem;}}
.bench-hero p {{color: {INK2}; margin: 0; max-width: 1050px; line-height: 1.5;}}
.bench-chips {{margin-top: 10px;}}
.bench-chip {{display: inline-block; padding: 3px 10px; margin: 0 6px 6px 0; border-radius: 999px;
    background: #f0efec; color: {INK2}; font-size: 0.8rem; border: 1px solid {BORDER};}}
.bench-note {{background: {SURFACE}; border: 1px solid {BORDER}; border-left: 3px solid {BLUE}; border-radius: 8px;
    padding: 10px 14px; color: {INK2}; margin: 6px 0 12px 0; line-height: 1.5;}}
.bench-legend {{display: inline-flex; align-items: center; margin-right: 16px; color: {INK2}; font-size: 0.85rem;}}
.bench-swatch {{width: 12px; height: 12px; border-radius: 3px; display: inline-block; margin-right: 6px;}}
</style>
"""


def _patch_plotly_chart():
    """Every chart uses our template (not Streamlit's), without the Plotly logo, and facet titles read 'x' not 'k=x'."""
    from streamlit.delta_generator import DeltaGenerator
    if getattr(DeltaGenerator, "_bench_patched", False):
        return
    orig = DeltaGenerator.plotly_chart

    def plotly_chart(self, fig, *a, **k):
        k.setdefault("theme", None)
        k.setdefault("config", {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})
        try:
            def facet(an):
                t = an.text or ""
                if "=" in t and "<" not in t:
                    key, _, val = t.partition("=")
                    if "budget" in key.lower() and not val.endswith("%"):
                        try:
                            val = f"{float(val):g}%"
                        except ValueError:
                            pass
                    an.update(text=val)
            fig.for_each_annotation(facet)
            fig.update_layout(legend_title_text="")
            if not any(t.showlegend is not False and t.name for t in fig.data) or fig.layout.showlegend is False:
                fig.update_layout(margin=dict(b=56))            # no legend: no room for one
        except Exception:                                     # noqa: BLE001
            pass
        return orig(self, fig, *a, **k)

    DeltaGenerator.plotly_chart = plotly_chart
    DeltaGenerator._bench_patched = True
    st.plotly_chart = lambda fig, *a, **k: plotly_chart(st._main, fig, *a, **k)


def page(title: str, icon: str):
    st.set_page_config(page_title=title, page_icon=icon, layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    _patch_plotly_chart()


def hero(eyebrow: str, title: str, text: str, chips: list[str] = ()):
    chip_html = "".join(f'<span class="bench-chip">{c}</span>' for c in chips)
    st.markdown(f'<div class="bench-hero"><div class="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{text}</p>'
                f'<div class="bench-chips">{chip_html}</div></div>', unsafe_allow_html=True)


def note(text: str):
    st.markdown(f'<div class="bench-note">{text}</div>', unsafe_allow_html=True)


def legend_html(items: list[tuple[str, str]]) -> str:
    return "".join(f'<span class="bench-legend"><span class="bench-swatch" style="background:{c}"></span>{t}</span>'
                   for t, c in items)



# models and devices keep the same colour on every chart and under every filter
MODEL_COLORS = {"qwen3-8b-fp16-ov": BLUE, "qwen3-4b-fp16-ov": ORANGE, "phi-3.5-mini-int4": AQUA}
DEVICE_COLORS = {"CPU": BLUE, "GPU": ORANGE, "NPU": AQUA, "HETERO:CPU,GPU,NPU": FULL}
STATUS_COLORS = {"SUCCESS": "#0ca30c", "FAILED": "#d03b3b"}


def color_map(column: str, values) -> dict:
    """Fixed colour per value of `column`: known models / devices / statuses, otherwise the palette in sorted order."""
    known = {"model": MODEL_COLORS, "device": DEVICE_COLORS, "status": STATUS_COLORS, "outcome": STATUS_COLORS}.get(column, {})
    vals = sorted({str(v) for v in values if v == v})
    out, i = {}, 0
    for v in vals:
        if v in known:
            out[v] = known[v]
        else:
            out[v] = SERIES[i % len(SERIES)]
            i += 1
    return out


def stat_row(items: list[tuple[str, str]]):
    """A row of stat tiles (label, value)."""
    cols = st.columns(len(items))
    for c, (label, value) in zip(cols, items):
        c.metric(label, value)

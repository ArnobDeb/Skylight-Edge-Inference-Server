"""Shared look and data loaders for the three dashboards (app.py, app_detailed.py, app_experiment3.py).

Look: one light theme (Streamlit config in .streamlit/config.toml + a Plotly template registered here), one colour
per role used everywhere: full cache = neutral grey, OpenVINO eviction = orange, + RestoreKV = blue, the hybrid
protocol = aqua. The categorical slots are a colour-blind-checked palette (adjacent pairs and the first three slots
pass the deutan / protan / tritan separation checks).

Data: reads the run folders under results/ live, in either folder layout (working copy or published copy).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]

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

pio.templates["restorekv"] = go.layout.Template(
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
pio.templates.default = "plotly_white+restorekv"

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
.rkv-hero {{padding: 4px 0 14px 0; border-bottom: 1px solid {GRID}; margin-bottom: 14px;}}
.rkv-hero .eyebrow {{font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; color: {BLUE};
    font-weight: 650;}}
.rkv-hero h1 {{margin: 2px 0 6px 0; padding: 0; font-size: 2.05rem;}}
.rkv-hero p {{color: {INK2}; margin: 0; max-width: 1050px; line-height: 1.5;}}
.rkv-chips {{margin-top: 10px;}}
.rkv-chip {{display: inline-block; padding: 3px 10px; margin: 0 6px 6px 0; border-radius: 999px;
    background: #f0efec; color: {INK2}; font-size: 0.8rem; border: 1px solid {BORDER};}}
.rkv-note {{background: {SURFACE}; border: 1px solid {BORDER}; border-left: 3px solid {BLUE}; border-radius: 8px;
    padding: 10px 14px; color: {INK2}; margin: 6px 0 12px 0; line-height: 1.5;}}
.rkv-legend {{display: inline-flex; align-items: center; margin-right: 16px; color: {INK2}; font-size: 0.85rem;}}
.rkv-swatch {{width: 12px; height: 12px; border-radius: 3px; display: inline-block; margin-right: 6px;}}
</style>
"""


def _patch_plotly_chart():
    """Every chart uses our template (not Streamlit's), without the Plotly logo, and facet titles read 'x' not 'k=x'."""
    from streamlit.delta_generator import DeltaGenerator
    if getattr(DeltaGenerator, "_rkv_patched", False):
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
    DeltaGenerator._rkv_patched = True
    st.plotly_chart = lambda fig, *a, **k: plotly_chart(st._main, fig, *a, **k)


def page(title: str, icon: str):
    st.set_page_config(page_title=title, page_icon=icon, layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    _patch_plotly_chart()


def hero(eyebrow: str, title: str, text: str, chips: list[str] = ()):
    chip_html = "".join(f'<span class="rkv-chip">{c}</span>' for c in chips)
    st.markdown(f'<div class="rkv-hero"><div class="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{text}</p>'
                f'<div class="rkv-chips">{chip_html}</div></div>', unsafe_allow_html=True)


def note(text: str):
    st.markdown(f'<div class="rkv-note">{text}</div>', unsafe_allow_html=True)


def legend_html(items: list[tuple[str, str]]) -> str:
    return "".join(f'<span class="rkv-legend"><span class="rkv-swatch" style="background:{c}"></span>{t}</span>'
                   for t, c in items)


# ---------------------------------------------------------------------------------------------------------- data
def res(*names: str) -> Path:
    """First existing results/<name> folder (private name first, then the published one)."""
    for n in names:
        p = ROOT / "results" / n
        if p.exists():
            return p
    return ROOT / "results" / names[0]


STAGE3 = lambda: res("stage3", "eviction_study")      # noqa: E731
STAGE4 = lambda: res("stage4", "genai_integration")   # noqa: E731

PROTOCOLS = {   # key -> (label, short label, colour)
    "proto": ("Python prototype · question-blind", "Prototype (question-blind)", FULL),
    "blind": ("GenAI · question-blind, inside order", "Question-blind · inside order", BLUE),
    "exact": ("GenAI · question-blind, exact order", "Question-blind · exact order", "#184f95"),   # same family, darker
    "hybrid": ("GenAI · question-blind, then continuous", "Blind → continuous (hybrid)", AQUA),
    "aware": ("GenAI · question-aware (default)", "Question-aware (GenAI default)", ORANGE),
}
POLICY_COLORS = {"H2O": BLUE, "ADAPTIVE_RKV": AQUA, "KVzip": YELLOW, "per-head": "#9a9890"}
MODELS = {"qwen3-8b": "Qwen3-8B", "qwen3-4b": "Qwen3-4B", "llama3.1-8b": "Llama-3.1-8B"}


def protocol_of(name: str) -> str:
    return ("exact" if "_qblind_exact" in name else "hybrid" if "hybrid" in name else
            "blind" if "_qblind" in name else "aware")


def model_of(name: str) -> str:
    return next((m for m in ("qwen3-4b", "llama3.1-8b") if m in name), "qwen3-8b")


def _jsonl(path: Path) -> pd.DataFrame:
    if not path.exists() or not path.stat().st_size:
        return pd.DataFrame()
    return pd.DataFrame([json.loads(x) for x in path.read_text().splitlines() if x.strip()])


@st.cache_data(ttl=60)
def genai_ruler() -> pd.DataFrame:
    """All GenAI RULER-4K rows: idx, task, variant (ov / ov+R / full), budget, score, protocol, model."""
    frames = []
    for d in sorted(STAGE4().glob("*genai_ruler4k*")):
        if any(s in d.name for s in ("smoke", "check")):
            continue
        df = _jsonl(d / "results.jsonl")
        if df.empty:
            continue
        df["protocol"], df["model"], df["run"] = protocol_of(d.name), model_of(d.name), d.name
        frames.append(df[["idx", "task", "variant", "budget", "score", "protocol", "model", "run"]])
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["budget"] = df["budget"].astype(float).round(3)
    return df.drop_duplicates(["idx", "variant", "budget", "protocol", "model"], keep="last")


@st.cache_data(ttl=300)
def prototype_ruler() -> pd.DataFrame:
    """The prototype's 130-sample OpenVINO H2O run (question-blind), same columns as genai_ruler()."""
    for d in sorted(STAGE3().glob("*ruler130_ovblock5000_block16")):
        df = _jsonl(d / "results.jsonl")
        if df.empty:
            continue
        df = df[df["method"].isin(["ov", "ov+restore", "full"])].copy()
        df["variant"] = df["method"].map({"ov": "ov", "ov+restore": "ov+R", "full": "full"})
        df["budget"] = df["budget"].astype(float).round(3)
        df["protocol"], df["model"], df["run"] = "proto", "qwen3-8b", d.name
        return df[["idx", "task", "variant", "budget", "score", "protocol", "model", "run"]]
    return pd.DataFrame()


@st.cache_data(ttl=60)
def paired_table(df: pd.DataFrame, n_boot: int = 10000, seed: int = 0) -> pd.DataFrame:
    """Per (protocol, model, budget): eviction and + RestoreKV means on the samples both have, the paired gain with a
    95% bootstrap interval, and the number of samples better / worse (same method as the report's figures)."""
    rows = []
    full = (df[df["variant"] == "full"].groupby(["protocol", "model"])["score"].mean()
            if len(df) else pd.Series(dtype=float))
    rngs = {}
    for (p, m, b), g in df[df["variant"] != "full"].groupby(["protocol", "model", "budget"]):
        rng = rngs.setdefault((p, m), np.random.default_rng(seed))    # one stream per run, as in the report
        w = g.pivot_table(index="idx", columns="variant", values="score")
        if not {"ov", "ov+R"} <= set(w.columns):
            continue
        w = w.dropna()
        if w.empty:
            continue
        diff = (w["ov+R"] - w["ov"]).to_numpy()
        boots = rng.choice(diff, (n_boot, len(diff))).mean(axis=1)
        fm = full.get((p, m), np.nan)
        if np.isnan(fm) and p != "proto":          # full cache does not depend on the protocol
            fm = full.xs(m, level="model").mean() if m in full.index.get_level_values("model") else np.nan
        rows.append({"protocol": p, "model": m, "budget": b, "n": len(w), "ov": w["ov"].mean(),
                     "ov+R": w["ov+R"].mean(), "gain": diff.mean(), "lo": np.percentile(boots, 2.5),
                     "hi": np.percentile(boots, 97.5), "better": int((diff > 0).sum()),
                     "worse": int((diff < 0).sum()), "full": fm})
    return pd.DataFrame(rows)


@st.cache_data(ttl=60)
def decode_scaling() -> pd.DataFrame:
    """Experiment 3A rows; `protocol` = aware (GenAI default) or hybrid (question-blind, then continuous)."""
    frames = []
    for d in sorted(STAGE4().glob("*_decode_scaling_*")):
        f = d / "results.csv"
        if not f.exists() or "_smoke" in d.name:
            continue
        x = pd.read_csv(f)
        x["protocol"] = "hybrid" if d.name.endswith("_hybrid") else "aware"
        frames.append(x)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df[df["status"].astype(str).str.startswith(("ok", "skipped"))]
    return df.drop_duplicates(["model", "config", "output_tokens", "batch", "protocol"], keep="last")


def fmt_tokens(v) -> str:
    v = int(v)
    return f"{v // 1024}K" if v >= 1024 and v % 1024 == 0 else str(v)


def pct(b: float) -> str:
    return f"{100 * float(b):g}%"

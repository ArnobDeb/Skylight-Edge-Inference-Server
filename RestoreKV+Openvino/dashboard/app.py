"""RestoreKV on OpenVINO: results dashboard.

Shows what RestoreKV adds on top of OpenVINO's own KV-cache eviction, at the same memory budget.
Reads the result folders under results/ live, so runs that are still going appear when the page is refreshed.

    conda activate arnob
    cd RestoreKV+Openvino && env -u PYTHONPATH streamlit run dashboard/app.py
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
KV_BYTES_PER_TOKEN = 36 * 2 * 8 * 128 * 2          # Qwen3-8B f16 KV: layers x (K, V) x KV heads x head dim x 2 bytes

GREY, GREEN = "#9aa5b1", "#2e7d32"
POLICY = {"ov": "OpenVINO H2O (default policy)", "arkv": "OpenVINO ADAPTIVE_RKV",
          "kvzip": "KVzip (the paper's eviction)", "ov_pht": "Per-head tokens (not in OpenVINO, what-if)"}
LADDER = {"kvzip": "KVzip, per head", "kvzip_layer": "KVzip, same count per layer", "kvzip_hs": "KVzip, all heads share tokens",
          "kvzip_block": "KVzip scores, OpenVINO block layout", "ov": "OpenVINO H2O"}
FOLDERS = {"ov_prototype": ["ov_prototype"], "adapter_training": ["adapter_training"], "eviction_study": ["eviction_study"]}

st.set_page_config(page_title="RestoreKV on OpenVINO", page_icon="📊", layout="wide")


# ---------------------------------------------------------------------------- data
def run_dirs(folder: str, pattern: str) -> list[Path]:
    out = []
    for name in FOLDERS[folder]:
        out += sorted((RES / name).glob(pattern))
    return out


@st.cache_data(ttl=60)
def load(folder: str, pattern: str) -> pd.DataFrame:
    frames = []
    for d in run_dirs(folder, pattern):
        f = d / "results.jsonl"
        if f.exists() and f.stat().st_size:
            df = pd.DataFrame([json.loads(l) for l in f.read_text().splitlines() if l.strip()])
            df["run"] = d.name
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["budget_pct"] = (df["budget"].fillna(1.0) * 100).round(1)
    df["base"] = df["method"].str.split("+").str[0]
    df["RestoreKV"] = df["method"].str.contains(r"\+restore").map({True: "with RestoreKV", False: "without"})
    df["kv_mb"] = (df["kept_context_frac"].fillna(1.0) * df["context_tokens"] + df.get("restore_entries", 0)) \
        * KV_BYTES_PER_TOKEN / 1e6
    return df.drop_duplicates(subset=["idx", "method", "budget_pct"])


def task_mean(df: pd.DataFrame) -> float:
    """RULER / LOFT convention: average per task (dataset), then over tasks."""
    return float(df.groupby("task")["score"].mean().mean()) if len(df) else float("nan")


def table(df: pd.DataFrame, label_map: dict) -> pd.DataFrame:
    rows = []
    for (base, b, r), g in df[df.method != "full"].groupby(["base", "budget_pct", "RestoreKV"]):
        rows.append({"policy": label_map.get(base, base), "budget": f"{b:g}%", "budget_pct": b, "RestoreKV": r,
                     "score": task_mean(g), "samples": g["idx"].nunique(), "kv_mb": g["kv_mb"].mean()})
    return pd.DataFrame(rows)


def paired(df: pd.DataFrame, base: str, b: float) -> tuple[int, int]:
    a = df[(df.method == base) & (df.budget_pct == b)].set_index("idx")["score"]
    c = df[(df.method == base + "+restore") & (df.budget_pct == b)].set_index("idx")["score"]
    d = (c - a).dropna()
    return int((d > 0).sum()), int((d < 0).sum())


def bars(t: pd.DataFrame, facet: str | None = "policy", height: int = 430, full: float | None = None):
    order = sorted(t.budget_pct.unique())
    fig = px.bar(t, x="budget", y="score", color="RestoreKV", barmode="group", facet_col=facet, text_auto=".1f",
                 category_orders={"budget": [f"{b:g}%" for b in order], "RestoreKV": ["without", "with RestoreKV"]},
                 color_discrete_map={"without": GREY, "with RestoreKV": GREEN}, height=height,
                 labels={"score": "score (0-100)", "budget": "budget (share of the context's KV cache kept)"})
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_traces(textposition="outside", textfont_size=11)
    if full is not None:
        fig.add_hline(y=full, line_dash="dash", annotation_text=f"full cache {full:.1f}")
    return fig


def gains_text(df: pd.DataFrame, t: pd.DataFrame, label_map: dict) -> str:
    inv = {v: k for k, v in label_map.items()}
    parts = []
    for pol, g in t.groupby("policy"):
        w = g.pivot_table(index="budget_pct", columns="RestoreKV", values="score")
        if {"without", "with RestoreKV"} <= set(w.columns):
            gain = (w["with RestoreKV"] - w["without"]).dropna()
            if len(gain):
                b = gain.idxmax()
                better, worse = paired(df, inv[pol], b)
                parts.append(f"**{pol}**: {gain.min():+.1f} to {gain.max():+.1f} points; "
                             f"best {gain.max():+.1f} at {b:g}% ({better} samples better, {worse} worse)")
    return "  \n".join(parts)


main = pd.concat([x for x in (load("eviction_study", "*ruler130_ovblock5000*"), load("eviction_study", "*ruler130_arkv2000*"),
                              load("eviction_study", "*ruler130_kvzip*"),
                              load("ov_prototype", "*_ruler4k_ladder_n10").query("base in ['kvzip', 'full']"))
                  if len(x)], ignore_index=True).drop_duplicates(subset=["idx", "method", "budget_pct"])
pre39 = pd.concat([x for x in (load("eviction_study", "*ovblock5000_ext_block16*"), load("eviction_study", "*arkv2000_ext_block16*"))
                   if len(x)], ignore_index=True)
ladder = load("ov_prototype", "*_ruler4k_ladder_n10")

# ---------------------------------------------------------------------------- page
st.title("RestoreKV on OpenVINO")
st.caption("Qwen3-8B on an Intel Core Ultra 7 258V laptop (Arc iGPU). Budget = share of the document's KV cache that is "
           "kept. RestoreKV's 8 summary slots come out of the same budget, so both bars of a pair use the same memory.")
if st.sidebar.button("Reload data"):
    st.cache_data.clear()
st.sidebar.markdown("Each tab has its own controls at the top. Data refreshes every minute.")

tabs = st.tabs(["Baseline vs RestoreKV", "Memory & speed", "Long context (32K)", "Why OpenVINO gains less", "Samples"])

# 1 --------------------------------------------------------------------------------
with tabs[0]:
    st.markdown("OpenVINO's built-in eviction **without** and **with** RestoreKV on RULER-4K (13 question types).")
    c = st.columns([1, 2, 1.4])
    sset = c[0].radio("Samples", ["130 (10 per task)", "39 (3 per task)"])
    src = main if sset.startswith("130") else pre39
    pols = [k for k in POLICY if k in set(src.base)]
    chosen = c[1].multiselect("Eviction policies", pols, default=[p for p in pols if p != "ov_pht"],
                              format_func=lambda k: POLICY[k])
    view = c[2].radio("View", ["each policy: without vs with", "all policies, with RestoreKV"])
    d = src[src.base.isin(chosen) | (src.method == "full")]
    if not chosen or not len(d):
        st.info("Pick at least one policy.")
    else:
        t = table(d, POLICY)
        full = task_mean(src[src.method == "full"])
        if view.startswith("each"):
            st.plotly_chart(bars(t, full=full), width="stretch")
        else:
            w = t[t.RestoreKV == "with RestoreKV"]
            fig = px.bar(w, x="budget", y="score", color="policy", barmode="group", text_auto=".1f", height=430,
                         category_orders={"budget": [f"{b:g}%" for b in sorted(w.budget_pct.unique())]})
            fig.update_traces(textposition="outside")
            fig.add_hline(y=full, line_dash="dash", annotation_text=f"full cache {full:.1f}")
            st.plotly_chart(fig, width="stretch")
        st.markdown(gains_text(d, t, POLICY))
        st.caption(f"{d['idx'].nunique()} samples. One sample changing its answer moves a bar by about "
                   f"{100 / max(1, d['idx'].nunique()):.1f} points.")

# 2 --------------------------------------------------------------------------------
with tabs[1]:
    st.markdown("**Memory**: the KV cache kept at each budget (the same for every policy). **Speed**: decode "
                "tokens per second with the cache physically cut to the budget.")
    c = st.columns(2)
    T = c[0].radio("Document length", [3860, 32768, 65536], format_func=lambda n: f"{n // 1000}K tokens", horizontal=True)
    mem = pd.DataFrame([{"budget": "full" if b == 100 else f"{b:g}%", "KV cache (MB)": T * b / 100 * KV_BYTES_PER_TOKEN / 1e6}
                        for b in (100, 25, 15, 10, 5, 2.5)])
    fig = px.bar(mem, x="budget", y="KV cache (MB)", text_auto=".0f", height=360, color_discrete_sequence=["#4a6d8c"])
    fig.update_traces(textposition="outside")
    c[0].plotly_chart(fig, width="stretch")
    thr_files = [p for d in run_dirs("eviction_study", "*_throughput_memory") for p in [d / "throughput.csv"] if p.exists()]
    if thr_files:
        thr = pd.concat([pd.read_csv(p) for p in thr_files], ignore_index=True)
        thr = thr[thr.status == "ok"].copy()
        thr["budget"] = thr["budget"].map(lambda b: "full" if pd.isna(b) else f"{100 * b:g}%")
        thr["setup"] = thr["config"].map({"full": "full cache", "ov": "eviction", "ov+restore": "eviction + RestoreKV"})
        ctx = c[1].radio("Speed measured on", sorted(thr.context.unique()), horizontal=True)
        batch = c[1].radio("Sequences decoded together", sorted(thr.batch.unique()), horizontal=True)
        s = thr[(thr.context == ctx) & (thr.batch == batch)]
        order = ["full"] + [f"{b:g}%" for b in (2.5, 5, 10, 15, 25)]
        fig = px.bar(s, x="budget", y="decode_tokens_per_s", color="setup", barmode="group", text_auto=".1f", height=360,
                     category_orders={"budget": [o for o in order if o in set(s.budget)]},
                     color_discrete_map={"full cache": "#1b2733", "eviction": GREY, "eviction + RestoreKV": GREEN},
                     labels={"decode_tokens_per_s": "decode tokens / s"})
        fig.update_traces(textposition="outside")
        c[1].plotly_chart(fig, width="stretch")
        c[1].caption("Missing bars did not fit in memory. RestoreKV keeps the same cache size, so its speed equals plain eviction's.")
    costs = []
    for d in run_dirs("eviction_study", "*_throughput_memory"):
        costs += [json.loads(f.read_text()) for f in sorted(d.glob("one_time_costs_*.json"))]
    if costs:
        cdf = pd.DataFrame(costs).set_index("context")
        cols = {"prefill_s": "read the document (prefill)", "score_openvino_h2o_s": "OpenVINO H2O scoring",
                "score_kvzip_s": "KVzip scoring", "restore_pass_s": "RestoreKV restore pass"}
        st.markdown("**One-time cost per document (seconds)**")
        st.dataframe(cdf[[k for k in cols if k in cdf]].rename(columns=cols).round(2), width="stretch")

# 3 --------------------------------------------------------------------------------
with tabs[2]:
    st.markdown("OpenVINO H2O eviction without and with RestoreKV on 32K-token documents (INT8 weights). "
                "Runs are still adding samples; reload to update.")
    for name, pat, lab in (("LOFT-32K", "*loft32k_ov_ovblock5000*", "datasets"), ("RULER-32K", "*ruler32k_ov_ovblock5000*", "tasks")):
        d = load("eviction_study", pat)
        if not len(d):
            st.info(f"{name}: no results yet.")
            continue
        t = table(d, POLICY)
        st.subheader(f"{name}: {d['idx'].nunique()} samples over {d['task'].nunique()} {lab}")
        st.plotly_chart(bars(t, facet=None, height=360, full=task_mean(d[d.method == "full"])), width="stretch")

# 4 --------------------------------------------------------------------------------
with tabs[3]:
    st.markdown("The paper's eviction (KVzip) decides per head and per token; OpenVINO decides per 16-token block, the "
                "same for all heads. Each step towards OpenVINO's layout costs accuracy, and RestoreKV gains most when "
                "eviction is fine-grained. (130 samples, the paper's released adapter, 32-token blocks.)")
    if len(ladder):
        b = st.radio("Budget", sorted(x for x in ladder.budget_pct.unique() if x < 100), index=1,
                     format_func=lambda v: f"{v:g}%", horizontal=True)
        g = ladder[(ladder.budget_pct == b) & ladder.base.isin(LADDER)]
        rows = [{"layout": LADDER[k], "RestoreKV": r, "score": task_mean(x)} for (k, r), x in g.groupby(["base", "RestoreKV"])]
        t = pd.DataFrame(rows)
        fig = px.bar(t, x="layout", y="score", color="RestoreKV", barmode="group", text_auto=".1f", height=430,
                     category_orders={"layout": list(LADDER.values()), "RestoreKV": ["without", "with RestoreKV"]},
                     color_discrete_map={"without": GREY, "with RestoreKV": GREEN}, labels={"layout": ""})
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, width="stretch")

# 5 --------------------------------------------------------------------------------
with tabs[4]:
    st.markdown("Look at individual answers: pick a run, a task and a sample.")
    allruns = pd.concat([x for x in (main, pre39, load("eviction_study", "*32k_ov_ovblock5000*")) if len(x)], ignore_index=True)
    if len(allruns):
        c = st.columns(3)
        run = c[0].selectbox("Run", sorted(allruns.run.unique()))
        r = allruns[allruns.run == run]
        task = c[1].selectbox("Task", sorted(r.task.unique()))
        idx = c[2].selectbox("Sample", sorted(r[r.task == task].idx.unique(), key=str))
        s = r[(r.task == task) & (r.idx == idx)].sort_values(["budget_pct", "method"])
        st.markdown(f"Expected answer: `{s['answer'].iloc[0]}`")
        st.dataframe(s[["method", "budget_pct", "score", "pred"]], width="stretch", hide_index=True)

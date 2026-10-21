"""Experiment 3 dashboard: RestoreKV inside OpenVINO GenAI on the Arc iGPU.

    streamlit run dashboard/app_experiment3.py

3A decode scaling (4K input, 128-16K output; full cache vs OpenVINO eviction vs eviction + RestoreKV), 3B agreement
of the evicted outputs with the full-cache output, and RULER-4K accuracy in the same pipeline. Reads the run folders
under results/genai_integration/ (or results/genai_integration/ in the published copy) live.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/genai_integration"))
sys.path.insert(0, str(ROOT / "experiments/genai_integration"))
RES = next((ROOT / "results" / d for d in ("stage4", "genai_integration") if (ROOT / "results" / d).exists()),
           ROOT / "results/genai_integration")

st.set_page_config(page_title="Experiment 3 — RestoreKV in OpenVINO GenAI", page_icon="⚡", layout="wide")

CFG_NAME = {"full": "Full cache (GenAI baseline)", "ov": "OpenVINO eviction 10%", "ov+R": "Eviction 10% + RestoreKV"}
CFG_COLOR = {CFG_NAME["full"]: "#5b6b7b", CFG_NAME["ov"]: "#e0912f", CFG_NAME["ov+R"]: "#2a9d55"}
METRIC = {
    "ttft_ms": "TTFT (ms)", "tpot_ms": "TPOT (ms/token)", "wall_tok_s": "Throughput (tokens/s, wall clock)",
    "throughput_tok_s": "Throughput (tokens/s, GenAI)", "generate_duration_ms": "Generation time (ms)",
    "inference_duration_ms": "Inference time (ms)", "kv_mb_per_seq_end": "KV cache held at the end (MB / sequence)",
    "kv_mb_full_cache_equiv": "KV cache a full cache would hold (MB)", "peak_gpu_gtt_gb": "Process GPU memory, peak (GB)",
    "peak_rss_mb": "Process RSS, peak (MB)", "peak_swap_mb": "Process swap, peak (MB)",
    "disk_read_mb": "Disk read during generation (MB)", "sys_pswpin": "System swap-in (pages)",
    "sys_pswpout": "System swap-out (pages)", "sys_pgmajfault": "Major page faults",
    "min_mem_available_gb": "Lowest free system memory (GB)",
}


def fmt_tokens(v):
    v = int(v)
    return f"{v // 1024}K" if v >= 1024 and v % 1024 == 0 else str(v)


def line(df, y, title, height=430, log_y=False):
    d = df.sort_values("output_tokens")
    fig = px.line(d, x="out_label", y=y, color="Configuration", markers=True, color_discrete_map=CFG_COLOR,
                  facet_col="model" if d["model"].nunique() > 1 else None, log_y=log_y,
                  labels={"out_label": "Generated output tokens", y: METRIC.get(y, y)}, title=title)
    fig.update_xaxes(type="category")
    fig.update_layout(height=height, hovermode="x unified", legend_title_text="", margin=dict(t=70, l=40, r=20))
    return fig


@st.cache_data(ttl=60)
def load_decode():
    frames = []
    for d in sorted(RES.glob("*_decode_scaling_*")):
        f = d / "results.csv"
        if f.exists() and "_smoke" not in d.name:
            frames.append(pd.read_csv(f))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df[df["status"].astype(str).str.startswith(("ok", "skipped"))].drop_duplicates(
        ["model", "config", "output_tokens", "batch"], keep="last")
    df["Configuration"] = df["config"].map(CFG_NAME)
    df["out_label"] = df["output_tokens"].map(fmt_tokens)
    return df


@st.cache_data(ttl=60)
def load_agreement():
    from output_agreement import compare   # noqa: E402
    rows = []
    for d in sorted(RES.glob("*_decode_scaling_*")):
        f = d / "generations.jsonl"
        if not f.exists() or "_smoke" in d.name:
            continue
        model = d.name.split("_decode_scaling_")[1]
        gens = {}
        for line_ in open(f):
            g = json.loads(line_)
            if g["batch"] == 1:
                gens[(g["config"], g["output_tokens"])] = g["ids"]
        for (cfg, out), ids in gens.items():
            if cfg != "full" and ("full", out) in gens:
                rows.append({"model": model, "config": cfg, "output_tokens": out, **compare(gens[("full", out)], ids)})
    df = pd.DataFrame(rows)
    if len(df):
        df["Configuration"] = df["config"].map(CFG_NAME)
        df["out_label"] = df["output_tokens"].map(fmt_tokens)
    return df


@st.cache_data(ttl=60)
def load_accuracy():
    rows = []
    for d in sorted(RES.glob("*genai_ruler4k*")):
        f = d / "summary.csv"
        if not f.exists() or "smoke" in d.name:
            continue
        name = d.name.split("_", 1)[1]
        protocol = ("question-blind, exact order" if "exact" in name else "question-blind" if "qblind" in name
                    else "question-aware (GenAI default)")
        model = "qwen3-8b"
        for m in ("qwen3-4b", "llama3.1-8b"):
            if m in name:
                model = m
        s = pd.read_csv(f)
        s["protocol"], s["model"] = protocol, model
        rows.append(s)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


@st.cache_data(ttl=300)
def load_prototype():
    """The Python prototype's 130-sample OpenVINO H2O run (question-blind), for the protocol comparison."""
    for res in (ROOT / "results/eviction_study", ROOT / "results/eviction_study"):
        for d in sorted(res.glob("*ruler130_ovblock5000_block16")):
            s_ = pd.read_csv(d / "summary.csv")
            s_ = s_[s_["method"].isin(["ov", "ov+restore"])]
            return pd.DataFrame({"variant": s_["method"].map({"ov": "ov", "ov+restore": "ov+R"}),
                                 "budget": s_["budget_kept"].astype(float), "score": s_["avg_score"]})
    return pd.DataFrame()


df = load_decode()
st.title("Experiment 3 — RestoreKV inside OpenVINO GenAI (Arc iGPU)")
st.caption("Same design as openvino_benchmarking experiment 2 (fixed input, growing output), now with KV-cache eviction "
           "and RestoreKV inside the real ContinuousBatchingPipeline. Input: 4K tokens of a RULER-4K document; "
           "output 128 → 16K tokens (ignore_eos); OpenVINO H2O eviction keeps 10% of the input's KV cache and keeps "
           "evicting while generating. Speed / memory tabs: GenAI default (question-aware) eviction; the Accuracy tab "
           "shows both the question-aware and the question-blind protocol.")
if df.empty:
    st.info(f"No decode-scaling runs yet in {RES}.")
    st.stop()

models = sorted(df["model"].unique())
sel_models = st.sidebar.multiselect("Models", models, default=models)
sel_cfgs = st.sidebar.multiselect("Configurations", list(CFG_NAME.values()), default=list(CFG_NAME.values()))
d = df[df["model"].isin(sel_models) & df["Configuration"].isin(sel_cfgs)]
d1 = d[(d["batch"] == 1) & d["status"].eq("ok")]
st.sidebar.markdown(f"**Runs:** {len(d1)} points (batch 1), {int((d['batch'] == 8).sum())} batch-8 points")

tabs = st.tabs(["Overview", "Latency & throughput", "Memory", "Batching", "Output agreement (3B)", "Accuracy", "Raw data"])

with tabs[0]:
    st.subheader("Savings at the longest output, against the GenAI baseline")
    if d1.empty:
        st.info("No completed batch-1 points.")
    for m in sel_models:
        dm = d1[d1["model"] == m]
        if dm.empty:
            continue
        top = dm["output_tokens"].max()
        x = dm[dm["output_tokens"] == top].set_index("config")
        st.markdown(f"**{m}** — {fmt_tokens(top)} output tokens")
        cols = st.columns(4)
        if "full" in x.index:
            for i, c in enumerate([c for c in ("ov", "ov+R") if c in x.index]):
                kv_save = 1 - x.loc[c, "kv_mb_per_seq_end"] / x.loc["full", "kv_mb_per_seq_end"]
                tpot = x.loc[c, "tpot_ms"] / x.loc["full", "tpot_ms"] - 1
                cols[2 * i].metric(f"{CFG_NAME[c]}: KV cache", f"{x.loc[c, 'kv_mb_per_seq_end']:.0f} MB",
                                   f"{-kv_save:+.0%} vs full", delta_color="inverse")
                cols[2 * i + 1].metric(f"{CFG_NAME[c]}: TPOT", f"{x.loc[c, 'tpot_ms']:.0f} ms",
                                       f"{tpot:+.0%} vs full", delta_color="inverse")
    st.caption("Negative = lower than the full cache (better). RestoreKV keeps the same cache size as plain eviction "
               "(its 8 entries sit inside the budget); its benefit is accuracy, see the Accuracy and Output agreement tabs.")
    st.plotly_chart(line(d1, "kv_mb_per_seq_end", "KV cache held per sequence: capped by eviction, growing without it"),
                    width="stretch")

with tabs[1]:
    st.markdown("How fast is generation as the output grows? Eviction caps the cache, so the time per token stops "
                "growing with the sequence; TTFT includes reading the 4K-token prompt.")
    m1, m2 = st.columns(2)
    m1.plotly_chart(line(d1, "tpot_ms", "Time per output token"), width="stretch")
    m2.plotly_chart(line(d1, "wall_tok_s", "Throughput (wall clock)"), width="stretch")
    m3, m4 = st.columns(2)
    m3.plotly_chart(line(d1, "ttft_ms", "Time to first token"), width="stretch")
    m4.plotly_chart(line(d1, "generate_duration_ms", "Generation time", log_y=True), width="stretch")

with tabs[2]:
    st.markdown("KV cache held at the end of generation vs what a full cache would hold, plus what the process and "
                "the system used. GenAI reserves its cache pool up front, so process GPU memory moves less than the "
                "cache itself; the cache actually held is the first chart.")
    a, b = st.columns(2)
    a.plotly_chart(line(d1, "kv_mb_per_seq_end", "KV cache held (MB / sequence)"), width="stretch")
    b.plotly_chart(line(d1, "peak_gpu_gtt_gb", "Process GPU memory, peak"), width="stretch")
    c_, e = st.columns(2)
    c_.plotly_chart(line(d1, "peak_rss_mb", "Process RSS, peak"), width="stretch")
    e.plotly_chart(line(d1, "min_mem_available_gb", "Lowest free system memory during the run"), width="stretch")
    with st.expander("Swap, page faults and disk reads (experiment-2 counters)"):
        for y in ("peak_swap_mb", "disk_read_mb", "sys_pswpin", "sys_pgmajfault"):
            st.plotly_chart(line(d1, y, METRIC[y], height=330), width="stretch")
        st.caption("System counters are system-wide deltas during the run; disk reads are this process's reads. Neither "
                   "attributes I/O to the KV cache.")

with tabs[3]:
    st.markdown("Eight requests at once: every decode step reads the weights once for all of them, and evicted caches "
                "are small, so more sequences fit and decode faster together.")
    db = d[d["output_tokens"].isin(d[d["batch"] == 8]["output_tokens"].unique())]
    if (db["batch"] == 8).any():
        db = db.assign(Batch=db["batch"].astype(str))
        fig = px.bar(db[db["status"].eq("ok")], x="Configuration", y="wall_tok_s", color="Batch", barmode="group",
                     facet_col="model", facet_row="out_label", color_discrete_sequence=["#9aa9b8", "#2a6f97"],
                     labels={"wall_tok_s": "Total tokens/s"}, title="Throughput, batch 1 vs batch 8")
        fig.update_layout(height=520)
        st.plotly_chart(fig, width="stretch")
        sk = db[db["status"].str.startswith("skipped")]
        if len(sk):
            st.caption("Not run (would not fit in memory): " + ", ".join(
                f"{r.model} {r.Configuration} batch {r.batch} {r.out_label}" for r in sk.itertuples()))
    else:
        st.info("No batch-8 points yet.")

with tabs[4]:
    ag = load_agreement()
    st.markdown("Same prompt, greedy decoding: how long does each evicted generation stay identical to the full-cache "
                "generation, and how much of its content overlaps? Higher = closer to the full model.")
    if ag.empty:
        st.info("No pairs yet.")
    else:
        ag = ag[ag["model"].isin(sel_models)]
        a, b = st.columns(2)
        a.plotly_chart(line(ag, "first_divergence", "Identical prefix (tokens before the first difference)"),
                       width="stretch")
        b.plotly_chart(line(ag, "4gram_f1", "4-gram overlap with the full-cache output (F1)"), width="stretch")
        st.dataframe(ag.drop(columns=["out_label"]).sort_values(["model", "output_tokens", "config"]), width="stretch")

with tabs[5]:
    acc = load_accuracy()
    st.markdown("""
**Two protocols — read them separately.** Both use the same eviction scorer (OpenVINO H2O), budget and adapter;
they differ only in *when* the cache is compressed:

| Protocol | When eviction happens | What it models | Comparable with |
|---|---|---|---|
| **Question-aware** (GenAI default, as shipped) | after the whole prompt, document **and question**, has been read; the scorer's last 8 tokens follow the question | one question per document, answered once | itself only |
| **Question-blind** (RestoreKV paper protocol) | after the **document**, before the question is seen; the question is read on the compressed cache | compress once, reuse the cache for many questions | the paper and the Python prototype |

Question-aware scores are much higher because the scorer already knows the question. The fair comparison inside each
protocol is *eviction vs eviction + RestoreKV* at the same budget. "Inside order" and "exact order" are two ways of
running the question-blind protocol in GenAI (restore tokens inside the prompt vs fed after the document prefill,
exactly as in the prototype).""")
    if acc.empty:
        st.info("No accuracy runs yet.")
    else:
        acc = acc[acc["model"].isin(sel_models)].copy()
        acc["Configuration"] = acc["variant"].map({"full": CFG_NAME["full"], "ov": "OpenVINO eviction",
                                                   "ov+R": "Eviction + RestoreKV"})
        acc["Budget"] = acc["budget"].apply(lambda b: "full" if pd.isna(b) else f"{float(b):.1%}")
        proto = load_prototype()
        if "qwen3-8b" in sel_models:
            st.subheader("Qwen3-8B: the same RULER-4K samples under every protocol")
            rows = []
            for b in sorted(acc[acc["variant"] != "full"]["budget"].dropna().unique()):
                row = {"Budget": f"{b:.1%}"}
                if len(proto):
                    pb = proto[proto["budget"].round(3) == round(b, 3)].set_index("variant")["score"]
                    row["Prototype, question-blind"] = f"{pb.get('ov', float('nan')):.1f} → {pb.get('ov+R', float('nan')):.1f}"
                for p_ in ("question-blind", "question-blind, exact order", "question-aware (GenAI default)"):
                    g_ = acc[(acc["model"] == "qwen3-8b") & (acc["protocol"] == p_) & (acc["budget"].round(3) == round(b, 3))]
                    g_ = g_.set_index("variant")["score"]
                    if len(g_):
                        label = {"question-blind": "GenAI, question-blind (inside order)",
                                 "question-blind, exact order": "GenAI, question-blind (exact order)",
                                 "question-aware (GenAI default)": "GenAI, question-aware (default)"}[p_]
                        row[label] = f"{g_.get('ov', float('nan')):.1f} → {g_.get('ov+R', float('nan')):.1f}"
                rows.append(row)
            st.dataframe(pd.DataFrame(rows).set_index("Budget"), width="stretch")
            st.caption("Each cell: OpenVINO eviction → eviction + RestoreKV. Full cache: 93.1 in both the prototype and GenAI.")
        for p_ in ("question-blind", "question-blind, exact order", "question-aware (GenAI default)"):
            sub = acc[acc["protocol"] == p_]
            if sub.empty:
                continue
            st.subheader({"question-blind": "Question-blind protocol (paper) — GenAI, inside order",
                          "question-blind, exact order": "Question-blind protocol (paper) — GenAI, exact prototype order",
                          "question-aware (GenAI default)": "Question-aware protocol — GenAI as shipped"}[p_])
            for m, g_ in sub.groupby("model"):
                ev = g_[g_["variant"] != "full"]
                if ev.empty:
                    continue
                fig = px.bar(ev, x="Budget", y="score", color="Configuration", barmode="group", text_auto=".1f",
                             color_discrete_map={"OpenVINO eviction": "#e0912f", "Eviction + RestoreKV": "#2a9d55"},
                             title=m)
                full = acc[(acc["model"] == m) & (acc["variant"] == "full")]
                if len(full):
                    fig.add_hline(y=float(full["score"].iloc[0]), line_dash="dash",
                                  annotation_text=f"full cache {float(full['score'].iloc[0]):.1f}")
                fig.update_layout(height=360, yaxis_title="RULER-4K score")
                st.plotly_chart(fig, width="stretch")

with tabs[6]:
    st.dataframe(d.drop(columns=["out_label"]), width="stretch")
    st.download_button("Download CSV", d.to_csv(index=False), "experiment3_decode_scaling.csv")

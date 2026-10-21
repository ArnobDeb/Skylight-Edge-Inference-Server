"""Experiment 3 dashboard: RestoreKV inside OpenVINO GenAI on the Arc iGPU.

    streamlit run dashboard/app_experiment3.py

3A decode scaling (4K input, 128-16K output; full cache vs OpenVINO eviction vs eviction + RestoreKV), 3B agreement
of the evicted outputs with the full-cache output, and RULER-4K accuracy in the same pipeline under every protocol.
Reads the run folders under results/genai_integration/ (or results/genai_integration/ in the published copy) live.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

ROOT = C.ROOT
sys.path.insert(0, str(ROOT / "experiments/genai_integration"))
sys.path.insert(0, str(ROOT / "experiments/genai_integration"))
RES = C.STAGE4()

C.page("Experiment 3 · RestoreKV in OpenVINO GenAI", "⚡")

CFG_NAME = {"full": "Full cache (GenAI baseline)", "ov": "OpenVINO eviction 10%", "ov+R": "Eviction 10% + RestoreKV"}
CFG_COLOR = {CFG_NAME["full"]: C.FULL, CFG_NAME["ov"]: C.EVICT, CFG_NAME["ov+R"]: C.RKV}
PROTO_NAME = {"aware": "Question-aware (GenAI default)", "hybrid": "Question-blind, then continuous (hybrid)"}
METRIC = {
    "ttft_ms": "TTFT (ms)", "tpot_ms": "TPOT (ms / token)", "wall_tok_s": "Throughput (tokens/s, wall clock)",
    "throughput_tok_s": "Throughput (tokens/s, GenAI)", "generate_duration_ms": "Generation time (ms)",
    "inference_duration_ms": "Inference time (ms)", "kv_mb_per_seq_end": "KV cache held at the end (MB / sequence)",
    "kv_mb_full_cache_equiv": "KV cache a full cache would hold (MB)", "peak_gpu_gtt_gb": "Process GPU memory, peak (GB)",
    "peak_rss_mb": "Process RSS, peak (MB)", "peak_swap_mb": "Process swap, peak (MB)",
    "disk_read_mb": "Disk read during generation (MB)", "sys_pswpin": "System swap-in (pages)",
    "sys_pswpout": "System swap-out (pages)", "sys_pgmajfault": "Major page faults",
    "min_mem_available_gb": "Lowest free system memory (GB)",
}
fmt_tokens = C.fmt_tokens


def line(df, y, title, height=420, log_y=False):
    d = df.sort_values("output_tokens")
    fig = px.line(d, x="out_label", y=y, color="Configuration", markers=True, color_discrete_map=CFG_COLOR,
                  category_orders={"Configuration": list(CFG_NAME.values())},
                  facet_col="Model" if d["Model"].nunique() > 1 else None, log_y=log_y,
                  labels={"out_label": "Generated output tokens", y: METRIC.get(y, y)}, title=title)
    fig.update_xaxes(type="category")
    fig.update_layout(height=height, hovermode="x unified")
    return fig


@st.cache_data(ttl=60)
def load_agreement():
    from output_agreement import compare   # noqa: E402
    rows = []
    for d in sorted(RES.glob("*_decode_scaling_*")):
        f = d / "generations.jsonl"
        if not f.exists() or "_smoke" in d.name:
            continue
        model = d.name.split("_decode_scaling_")[1].removesuffix("_hybrid")
        gens = {}
        for line_ in open(f):
            g = json.loads(line_)
            if g["batch"] == 1:
                gens[(g["config"], g["output_tokens"])] = g["ids"]
        for (cfg, out), ids in gens.items():
            if cfg != "full" and ("full", out) in gens:
                rows.append({"model": model, "protocol": "hybrid" if d.name.endswith("_hybrid") else "aware",
                             "config": cfg, "output_tokens": out, **compare(gens[("full", out)], ids)})
    df = pd.DataFrame(rows)
    if len(df):
        df["Configuration"] = df["config"].map(CFG_NAME)
        df["out_label"] = df["output_tokens"].map(fmt_tokens)
        df["Model"] = df["model"].map(C.MODELS).fillna(df["model"])
    return df


raw = C.decode_scaling()
C.hero("Experiment 3 · OpenVINO GenAI on the Arc iGPU", "RestoreKV inside the real GenAI pipeline",
       "The openvino_benchmarking experiment-2 design (fixed input, growing output), now with KV-cache eviction and "
       "RestoreKV inside the ContinuousBatchingPipeline. A 4K-token RULER-4K document goes in; 128 → 16K tokens come "
       "out (ignore_eos). OpenVINO H2O eviction caps the cache at 10% of the input and keeps evicting while "
       "generating; RestoreKV's 8 slots come out of that same cap.",
       ["Qwen3-8B · Qwen3-4B · Llama-3.1-8B", "fp16 weights, f16 KV", "batch 1 + batch 8", "greedy decoding"])
if raw.empty:
    st.info(f"No decode-scaling runs yet in {RES}.")
    st.stop()

raw["Configuration"] = raw["config"].map(CFG_NAME)
raw["out_label"] = raw["output_tokens"].map(fmt_tokens)
raw["Model"] = raw["model"].map(C.MODELS).fillna(raw["model"])
models = [m for m in C.MODELS.values() if m in set(raw["Model"])]

st.sidebar.markdown("### Filters")
sel_models = st.sidebar.multiselect("Models", models, default=models)
sel_cfgs = st.sidebar.multiselect("Configurations", list(CFG_NAME.values()), default=list(CFG_NAME.values()))
protos = [p for p in ("aware", "hybrid") if p in set(raw["protocol"])]
sel_proto = st.sidebar.radio("Eviction protocol (speed & memory tabs)", protos, format_func=PROTO_NAME.get,
                             help="Question-aware: GenAI as shipped. Hybrid: the document is compressed once before "
                                  "the request is read (paper protocol), then GenAI keeps evicting during the output. "
                                  "The full-cache reference is the same run in both.") if len(protos) > 1 else "aware"
df = raw[(raw["protocol"] == sel_proto) | (raw["config"] == "full")]
df = df[~((df["config"] == "full") & (df["protocol"] != "aware"))]
d = df[df["Model"].isin(sel_models) & df["Configuration"].isin(sel_cfgs)]
d1 = d[(d["batch"] == 1) & d["status"].eq("ok")]
st.sidebar.caption(f"{len(d1)} batch-1 points · {int((d['batch'] == 8).sum())} batch-8 points · refreshes every minute")
if st.sidebar.button("Reload data"):
    st.cache_data.clear()

tabs = st.tabs(["Overview", "Latency & throughput", "Memory", "Batching", "Output agreement (3B)", "Accuracy",
                "Raw data"])

with tabs[0]:
    st.subheader("At the longest output: what eviction saves against the GenAI baseline")
    if d1.empty:
        st.info("No completed batch-1 points.")
    for m in sel_models:
        dm = d1[d1["Model"] == m]
        if dm.empty:
            continue
        top = dm["output_tokens"].max()
        x = dm[dm["output_tokens"] == top].set_index("config")
        st.markdown(f"**{m}** · {fmt_tokens(top)} output tokens")
        cols = st.columns(4)
        if "full" in x.index:
            for i, c in enumerate([c for c in ("ov", "ov+R") if c in x.index]):
                kv_save = 1 - x.loc[c, "kv_mb_per_seq_end"] / x.loc["full", "kv_mb_per_seq_end"]
                tpot = x.loc[c, "tpot_ms"] / x.loc["full", "tpot_ms"] - 1
                cols[2 * i].metric(f"{CFG_NAME[c]} · KV cache", f"{x.loc[c, 'kv_mb_per_seq_end']:.0f} MB",
                                   f"{-kv_save:+.0%} vs full", delta_color="inverse")
                cols[2 * i + 1].metric(f"{CFG_NAME[c]} · TPOT", f"{x.loc[c, 'tpot_ms']:.0f} ms",
                                       f"{tpot:+.0%} vs full", delta_color="inverse")
    C.note("Negative = lower than the full cache, which is better here. RestoreKV keeps exactly the same cache size as "
           "plain eviction, because its 8 entries sit inside the budget. Its benefit is accuracy: see the Accuracy and "
           "Output agreement tabs. KV sizes are computed from the configuration (max_cache_size + one 16-token "
           "block); the measured device memory is in the Memory tab.")
    st.plotly_chart(line(d1, "kv_mb_per_seq_end", "KV cache held per sequence: capped by eviction, growing without it"),
                    width="stretch")

with tabs[1]:
    st.markdown("How fast is generation as the output grows? Eviction caps the cache, so the time per token stops "
                "growing with the sequence. TTFT includes reading the 4K-token prompt.")
    m1, m2 = st.columns(2)
    m1.plotly_chart(line(d1, "tpot_ms", "Time per output token"), width="stretch")
    m2.plotly_chart(line(d1, "wall_tok_s", "Throughput (wall clock)"), width="stretch")
    m3, m4 = st.columns(2)
    m3.plotly_chart(line(d1, "ttft_ms", "Time to first token"), width="stretch")
    m4.plotly_chart(line(d1, "generate_duration_ms", "Generation time (log scale)", log_y=True), width="stretch")

with tabs[2]:
    st.markdown("KV cache held at the end of generation, against what the process and the system used. GenAI "
                "reserves its cache pool up front and the fp16 weights dominate, so process GPU memory moves much "
                "less than the cache itself.")
    a, b = st.columns(2)
    a.plotly_chart(line(d1, "kv_mb_per_seq_end", "KV cache held (MB / sequence)"), width="stretch")
    b.plotly_chart(line(d1, "peak_gpu_gtt_gb", "Process GPU memory, peak (measured)"), width="stretch")
    c_, e = st.columns(2)
    c_.plotly_chart(line(d1, "peak_rss_mb", "Process RSS, peak"), width="stretch")
    e.plotly_chart(line(d1, "min_mem_available_gb", "Lowest free system memory during the run"), width="stretch")
    with st.expander("Swap, page faults and disk reads (experiment-2 counters)"):
        for y in ("peak_swap_mb", "disk_read_mb", "sys_pswpin", "sys_pgmajfault"):
            st.plotly_chart(line(d1, y, METRIC[y], height=320), width="stretch")
        st.caption("System counters are system-wide deltas during the run; disk reads are this process's reads. Neither "
                   "attributes I/O to the KV cache.")

with tabs[3]:
    st.markdown("Eight requests at once: every decode step reads the weights once for all of them, and evicted caches "
                "are small, so more sequences fit and decode faster together.")
    db = raw[raw["Model"].isin(sel_models) & raw["Configuration"].isin(sel_cfgs) & (raw["protocol"] == "aware")]
    db = db[db["output_tokens"].isin(db[db["batch"] == 8]["output_tokens"].unique())]
    if (db["batch"] == 8).any():
        ok = db[db["status"].eq("ok")].assign(Batch=lambda x: "batch " + x["batch"].astype(str))
        fig = px.bar(ok, x="Configuration", y="wall_tok_s", color="Batch", barmode="group", text_auto=".1f",
                     facet_col="Model", facet_row="out_label", color_discrete_sequence=["#b7d3f6", C.BLUE],
                     category_orders={"Configuration": list(CFG_NAME.values()), "Batch": ["batch 1", "batch 8"]},
                     labels={"wall_tok_s": "Total tokens/s", "Configuration": "", "out_label": "output"},
                     title="Throughput, batch 1 vs batch 8 (all sequences together)")
        fig.update_layout(height=560)
        fig.update_xaxes(tickangle=-20)
        st.plotly_chart(fig, width="stretch")
        sk = db[db["status"].str.startswith("skipped")]
        if len(sk):
            st.caption("Not run (would not fit in the 30 GB of shared memory): " + ", ".join(
                f"{r.Model} {r.Configuration} batch {r.batch} {r.out_label}" for r in sk.itertuples()))
    else:
        st.info("No batch-8 points yet.")

with tabs[4]:
    ag = load_agreement()
    st.markdown("Same prompt, greedy decoding: how long does each evicted generation stay identical to the full-cache "
                "generation, and how much of its content overlaps? Higher = closer to the full model.")
    if ag.empty:
        st.info("No pairs yet.")
    else:
        ag = ag[ag["Model"].isin(sel_models) & (ag["protocol"] == sel_proto)]
        a, b = st.columns(2)
        a.plotly_chart(line(ag, "first_divergence", "Identical prefix (tokens before the first difference)"),
                       width="stretch")
        b.plotly_chart(line(ag, "4gram_f1", "4-gram overlap with the full-cache output (F1)"), width="stretch")
        C.note("One prompt per model and output length, so read this as an illustration, not a measurement with an "
               "error bar.")
        st.dataframe(ag.drop(columns=["out_label", "Model"]).sort_values(["model", "output_tokens", "config"]),
                     width="stretch", hide_index=True)

with tabs[5]:
    st.markdown("""
**Read every protocol separately.** All use the same eviction scorer (OpenVINO H2O), budget and adapter; they differ
only in *when* the cache is compressed:

| Protocol | When eviction happens | What it models | Comparable with |
|---|---|---|---|
| **Question-aware** (GenAI default, as shipped) | after the whole prompt, document **and question**; the scorer's last 8 tokens are the question | one question per document, answered once | itself only |
| **Question-blind** (RestoreKV paper protocol) | after the **document**, before the question; the question is read on the compressed cache | compress once, reuse the cache for any question | the paper and the Python prototype |
| **Blind → continuous** (hybrid, new) | once after the document, then GenAI keeps evicting while the answer is written | compress once, then generate at length with a capped cache | question-blind |

*Inside order* and *exact order* are two ways of running the question-blind protocol in GenAI: restore tokens inside
the document prefill, or fed after it on the full cache exactly as in the prototype.""")
    ruler = pd.concat([x for x in (C.genai_ruler(), C.prototype_ruler()) if len(x)], ignore_index=True)
    if ruler.empty:
        st.info("No accuracy runs yet.")
    else:
        t = C.paired_table(ruler)
        t["Model"] = t["model"].map(C.MODELS)
        t = t[t["Model"].isin(sel_models)]
        if "Qwen3-8B" in sel_models:
            st.subheader("Qwen3-8B: the same 130 RULER-4K samples under every protocol")
            q = t[t["model"] == "qwen3-8b"]
            order = [p for p in ("proto", "blind", "exact", "hybrid", "aware") if p in set(q["protocol"])]
            cell = q.assign(txt=lambda x: x.apply(
                lambda r: f"{r['ov']:.1f} → {r['ov+R']:.1f}  ({r['gain']:+.1f} [{r['lo']:+.1f}, {r['hi']:+.1f}])"
                + ("" if r["n"] >= 130 else f" · {int(r['n'])} so far"), axis=1))
            show = cell.pivot_table(index="budget", columns="protocol", values="txt", aggfunc="first")[order]
            show.index = [C.pct(b) for b in show.index]
            show.columns = [C.PROTOCOLS[p][1] for p in order]
            st.dataframe(show, width="stretch")
            st.caption("Each cell: eviction → eviction + RestoreKV (paired gain [95% bootstrap interval]). Full cache "
                       "93.1 in both the prototype and GenAI. An interval that excludes 0 is a significant gain.")
            g = q[q["protocol"].isin(order)].copy()
            g["Protocol"] = g["protocol"].map(lambda p: C.PROTOCOLS[p][1])
            g["Budget"] = g["budget"].map(C.pct)
            g["err_hi"], g["err_lo"] = g["hi"] - g["gain"], g["gain"] - g["lo"]
            fig = px.bar(g, x="Budget", y="gain", color="Protocol", barmode="group", error_y="err_hi",
                         error_y_minus="err_lo", text_auto="+.1f",
                         category_orders={"Protocol": [C.PROTOCOLS[p][1] for p in order]},
                         color_discrete_map={C.PROTOCOLS[p][1]: C.PROTOCOLS[p][2] for p in order},
                         labels={"gain": "RestoreKV gain (points)", "Budget": "budget (share of the KV cache kept)"},
                         title="RestoreKV gain per protocol, with 95% intervals")
            fig.update_traces(error_y=dict(color=C.MUTED, thickness=1.2, width=3), textposition="outside")
            fig.add_hline(y=0, line_color=C.AXIS)
            fig.update_layout(height=430)
            st.plotly_chart(fig, width="stretch")
        st.subheader("Eviction vs eviction + RestoreKV, per protocol and model")
        for p in ("blind", "exact", "hybrid", "aware"):
            sub = t[t["protocol"] == p]
            if sub.empty:
                continue
            st.markdown(f"**{C.PROTOCOLS[p][0]}**")
            cols = st.columns(max(1, sub["Model"].nunique()))
            for col, (m, g_) in zip(cols, sub.groupby("Model", sort=False)):
                long = g_.melt(id_vars=["budget", "full"], value_vars=["ov", "ov+R"], var_name="v", value_name="score")
                long["Configuration"] = long["v"].map({"ov": "OpenVINO eviction", "ov+R": "Eviction + RestoreKV"})
                long["Budget"] = long["budget"].map(C.pct)
                fig = px.bar(long, x="Budget", y="score", color="Configuration", barmode="group", text_auto=".0f",
                             color_discrete_map={"OpenVINO eviction": C.EVICT, "Eviction + RestoreKV": C.RKV},
                             labels={"score": "RULER-4K score"}, title=m, range_y=[0, 100])
                fm = g_["full"].dropna()
                if len(fm):
                    fig.add_hline(y=float(fm.iloc[0]), line_dash="dash", line_color=C.FULL,
                                  annotation_text=f"full cache {float(fm.iloc[0]):.1f}", annotation_position="top left")
                fig.update_traces(textposition="outside")
                fig.update_layout(height=340)
                col.plotly_chart(fig, width="stretch")
        with st.expander("All accuracy numbers"):
            st.dataframe(t.drop(columns=["Model"]).round(2), width="stretch", hide_index=True)

with tabs[6]:
    st.dataframe(d.drop(columns=["out_label"]), width="stretch", hide_index=True)
    st.download_button("Download CSV", d.to_csv(index=False), "experiment3_decode_scaling.csv")

"""RestoreKV on OpenVINO: summary dashboard.

What RestoreKV adds on top of OpenVINO's own KV-cache eviction at the same memory budget: first in the Python
prototype, then inside the real OpenVINO GenAI pipeline (both evaluation protocols), plus the speed and memory of
long generations. Reads the result folders under results/ live, so runs still going appear on refresh.

    conda activate arnob
    cd RestoreKV+Openvino && env -u PYTHONPATH streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

RES = C.ROOT / "results"
LADDER = {"kvzip": "KVzip, per head", "kvzip_layer": "KVzip, same count per layer", "kvzip_hs": "KVzip, all heads share tokens",
          "kvzip_block": "KVzip scores, OpenVINO block layout", "ov": "OpenVINO H2O"}
FOLDERS = {"ov_prototype": ["ov_prototype"], "adapter_training": ["adapter_training"], "eviction_study": ["eviction_study"]}

C.page("RestoreKV on OpenVINO", "📊")


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
            df = pd.DataFrame([json.loads(x) for x in f.read_text().splitlines() if x.strip()])
            df["run"] = d.name
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["budget_pct"] = (df["budget"].fillna(1.0) * 100).round(1)
    df["base"] = df["method"].str.split("+").str[0]
    df["RestoreKV"] = df["method"].str.contains(r"\+restore").map({True: "with RestoreKV", False: "without"})
    return df.drop_duplicates(subset=["idx", "method", "budget_pct"])


def task_mean(df: pd.DataFrame) -> float:
    return float(df.groupby("task")["score"].mean().mean()) if len(df) else float("nan")


ruler = pd.concat([x for x in (C.genai_ruler(), C.prototype_ruler()) if len(x)], ignore_index=True)
paired = C.paired_table(ruler) if len(ruler) else pd.DataFrame()
dec = C.decode_scaling()

# ---------------------------------------------------------------------------------------------------------- header
C.hero("M.Tech thesis · learned KV-cache restoration", "RestoreKV on OpenVINO",
       "OpenVINO saves memory by evicting most of the KV cache. RestoreKV adds 8 learned summary tokens, written before "
       "eviction and never evicted, so the model can still answer from what was dropped. Their slots come out of the "
       "same budget: with and without RestoreKV, the cache has exactly the same size.",
       ["Intel Core Ultra 7 258V · Arc iGPU", "OpenVINO 2026.4.1 + patched GenAI", "RULER-4K, 130 samples",
        "Qwen3-8B (+ Qwen3-4B, Llama-3.1-8B)"])
if st.sidebar.button("Reload data"):
    st.cache_data.clear()
st.sidebar.caption("Data refreshes every minute. "
                   + ("The detailed dashboard (app_detailed.py) has every prototype run; "
                      if (C.ROOT / "dashboard/app_detailed.py").exists() else "")
                   + "app_genai.py has the GenAI decode-scaling runs.")


def pair_bars(q: pd.DataFrame, protos: list[str], height: int = 430):
    """Small multiples, one per protocol: OpenVINO eviction vs eviction + RestoreKV per budget, full cache dashed."""
    long = q[q.protocol.isin(protos)].melt(id_vars=["protocol", "budget", "full"], value_vars=["ov", "ov+R"],
                                          var_name="v", value_name="score")
    long["Protocol"] = long["protocol"].map(lambda p: C.PROTOCOLS[p][0])
    long["Configuration"] = long["v"].map({"ov": "OpenVINO eviction", "ov+R": "Eviction + RestoreKV"})
    long["Budget"] = long["budget"].map(C.pct)
    fig = px.bar(long, x="Budget", y="score", color="Configuration", barmode="group", facet_col="Protocol",
                 text_auto=".0f", category_orders={"Protocol": [C.PROTOCOLS[p][0] for p in protos]},
                 color_discrete_map={"OpenVINO eviction": C.EVICT, "Eviction + RestoreKV": C.RKV},
                 labels={"score": "RULER-4K score", "Budget": "budget"}, range_y=[0, 100])
    full = q.full.dropna()
    if len(full):
        fig.add_hline(y=float(full.iloc[0]), line_dash="dash", line_color=C.FULL,
                      annotation_text=f"full cache {float(full.iloc[0]):.1f}", annotation_position="top left")
    fig.update_traces(textposition="outside")
    fig.update_layout(height=height)
    return fig


def gain_of(p: str, b: float, m: str = "qwen3-8b"):
    r = paired[(paired.protocol == p) & (paired.model == m) & (paired.budget.round(3) == round(b, 3))] \
        if len(paired) else paired
    return r.iloc[0] if len(r) else None


k = st.columns(4)
if len(paired):
    for col, p, lab in ((k[0], "blind", "Question-blind (paper protocol)"), (k[1], "aware", "Question-aware (GenAI default)")):
        g = paired[(paired.protocol == p) & (paired.model == "qwen3-8b")]
        if len(g):
            best = g.loc[g.gain.idxmax()]
            col.metric(f"{lab}", f"{best.gain:+.1f} pts", f"{best.ov:.1f} → {best['ov+R']:.1f} at {C.pct(best.budget)}",
                       delta_color="off",
                       help=f"Best RestoreKV gain over plain OpenVINO eviction, Qwen3-8B, 130 samples. Significant at "
                            f"{int((g.lo > 0).sum())} of {len(g)} budgets (95% interval above 0).")
if len(dec):
    x = dec[(dec.model == "qwen3-8b") & (dec.batch == 1) & (dec.protocol == "aware") & dec.status.eq("ok")]
    if len(x):
        top = x[x.output_tokens == x.output_tokens.max()].set_index("config")
        if {"full", "ov+R"} <= set(top.index):
            kv, kvf = top.loc["ov+R", "kv_mb_per_seq_end"], top.loc["full", "kv_mb_per_seq_end"]
            k[2].metric(f"KV cache at {C.fmt_tokens(x.output_tokens.max())} output", f"{kv:.0f} MB",
                        f"{kv / kvf - 1:+.0%} vs {kvf:,.0f} MB full", delta_color="inverse",
                        help="Qwen3-8B, 4K input, eviction + RestoreKV (same cache as plain eviction).")
            k[3].metric("Time per token, + RestoreKV", f"{top.loc['ov+R', 'tpot_ms']:.0f} ms",
                        f"{top.loc['ov+R', 'tpot_ms'] / top.loc['full', 'tpot_ms'] - 1:+.0%} vs full cache",
                        delta_color="inverse", help="Same as plain eviction: RestoreKV adds no cost per token.")

tabs = st.tabs(["Main result", "Inside OpenVINO GenAI", "Across models", "Speed & memory", "Why the gain is smaller",
                "Samples"])

# ---------------------------------------------------------------------------------------------------------- 1
with tabs[0]:
    st.subheader("OpenVINO eviction without and with RestoreKV, same memory")
    C.note("<b>Two protocols, read separately.</b> <i>Question-blind</i> (the RestoreKV paper's protocol): the cache is "
           "compressed after the document, before the question is known, so it can be reused for any question. "
           "<i>Question-aware</i> (OpenVINO GenAI as shipped): the question is already in the prompt when eviction "
           "scores the cache, which makes eviction itself much stronger. Compare eviction vs + RestoreKV inside one "
           "protocol, never across.")
    if paired.empty:
        st.info("No RULER-4K results yet.")
    else:
        q = paired[paired.model == "qwen3-8b"]
        order = [p for p in ("proto", "blind", "exact", "hybrid", "aware") if p in set(q.protocol)]
        pick = st.multiselect("Protocols", order, default=[p for p in ("proto", "blind", "aware") if p in order],
                              format_func=lambda p: C.PROTOCOLS[p][0])
        if pick:
            st.plotly_chart(pair_bars(q, [p for p in order if p in pick]), width="stretch")

            g = q[q.protocol.isin(pick)].copy()
            g["Protocol"] = g["protocol"].map(lambda p: C.PROTOCOLS[p][0])
            fig = go.Figure()
            for p in [p for p in order if p in pick]:
                s = g[g.protocol == p].sort_values("budget")
                fig.add_scatter(x=s.budget.map(C.pct), y=s.gain, mode="lines+markers", name=C.PROTOCOLS[p][0],
                                line=dict(color=C.PROTOCOLS[p][2]), marker=dict(color=C.PROTOCOLS[p][2]),
                                error_y=dict(type="data", array=s.hi - s.gain, arrayminus=s.gain - s.lo,
                                             color=C.PROTOCOLS[p][2], thickness=1.2, width=4),
                                customdata=s[["ov", "ov+R", "lo", "hi", "better", "worse"]].to_numpy(),
                                hovertemplate="%{x}: %{y:+.1f} pts [%{customdata[2]:+.1f}, %{customdata[3]:+.1f}]"
                                              "<br>%{customdata[0]:.1f} → %{customdata[1]:.1f}"
                                              "<br>%{customdata[4]} better / %{customdata[5]} worse<extra></extra>")
            fig.add_hline(y=0, line_color=C.AXIS)
            fig.update_layout(title="RestoreKV gain (paired, per sample) with 95% bootstrap intervals", height=380,
                              xaxis_title="budget", yaxis_title="gain (points)", hovermode="closest")
            st.plotly_chart(fig, width="stretch")
            tbl = g.assign(Budget=g.budget.map(C.pct),
                           **{"eviction → + RestoreKV": g.apply(lambda r: f"{r.ov:.1f} → {r['ov+R']:.1f}", axis=1),
                              "gain [95% interval]": g.apply(lambda r: f"{r.gain:+.1f} [{r.lo:+.1f}, {r.hi:+.1f}]", axis=1),
                              "better / worse": g.apply(lambda r: f"{r.better} / {r.worse}", axis=1)})
            st.dataframe(tbl[["Protocol", "Budget", "eviction → + RestoreKV", "gain [95% interval]", "better / worse",
                              "n"]].rename(columns={"n": "samples"}), width="stretch", hide_index=True,
                         height=36 * (len(tbl) + 1) + 4)
            st.caption("130 samples (10 per RULER-4K task); one sample changing its answer moves a score by ~0.8 points. "
                       "Gain = mean of the per-sample differences; a 95% interval above 0 means the gain is significant.")

# ---------------------------------------------------------------------------------------------------------- 2
with tabs[1]:
    st.subheader("How RestoreKV runs inside OpenVINO GenAI")
    st.markdown("""
The adapter is exported into the OpenVINO model itself; the C++ pipeline only needs to know which token ids are the
restore tokens and keep their KV blocks. Changes in the GenAI fork (`restorekv-2026.4.1`, about 250 lines):

| Piece | Where | What it does |
|---|---|---|
| Restore tokens in the model | export script | ids V…V+7 (past the vocabulary) look up the 8 learned embeddings; the LoRA is gated to those ids; `rt_info` names them |
| Protection | `cache_eviction.cpp` | blocks holding restore tokens get the highest score, are never evicted, and are tracked as blocks move |
| Question-blind mode | `pipeline_impl.cpp`, `sampler.cpp` | the prompt is split at a separator: the document is prefilled and evicted once, the question is fed afterwards on the compressed cache |
| Exact order | `pipeline_impl.cpp` | the restore tokens run after the document on the full cache, their attention is kept out of the scores, eviction follows the 8th |
""")
    if len(paired):
        q = paired[paired.model == "qwen3-8b"]
        order = [p for p in ("proto", "blind", "exact", "hybrid") if p in set(q.protocol)]
        st.markdown("**The question-blind protocol in three implementations** (same 130 samples, same adapter)")
        st.plotly_chart(pair_bars(q, order, height=400), width="stretch")
        C.note("Plain eviction in GenAI reproduces the prototype within 1–2 points (same protocol), so the C++ port "
               "evicts what the prototype evicts. The exact order reproduces the prototype's RestoreKV gains; the "
               "inside order (restore tokens inside the document prefill) gains more at every budget.")

# ---------------------------------------------------------------------------------------------------------- 3
with tabs[2]:
    st.subheader("The same pipeline on other models (question-aware, GenAI default)")
    st.markdown("Each model has its own adapter, trained with the same recipe against OpenVINO's eviction "
                "(5,000 steps on an NVIDIA A40). RULER-4K, 130 samples.")
    if len(paired):
        a = paired[paired.protocol == "aware"].copy()
        a["Model"] = a["model"].map(C.MODELS)
        cols = st.columns(3)
        for col, (m, lab) in zip(cols, C.MODELS.items()):
            g = a[a.model == m]
            if g.empty:
                continue
            long = g.melt(id_vars=["budget", "full"], value_vars=["ov", "ov+R"], var_name="v", value_name="score")
            long["Configuration"] = long["v"].map({"ov": "OpenVINO eviction", "ov+R": "Eviction + RestoreKV"})
            long["Budget"] = long["budget"].map(C.pct)
            fig = px.bar(long, x="Budget", y="score", color="Configuration", barmode="group", text_auto=".0f",
                         color_discrete_map={"OpenVINO eviction": C.EVICT, "Eviction + RestoreKV": C.RKV},
                         labels={"score": "RULER-4K score"}, title=lab, range_y=[0, 100])
            if g.full.notna().any():
                fig.add_hline(y=float(g.full.dropna().iloc[0]), line_dash="dash", line_color=C.FULL,
                              annotation_text=f"full {float(g.full.dropna().iloc[0]):.1f}", annotation_position="top left")
            fig.update_traces(textposition="outside")
            fig.update_layout(height=360, showlegend=m == "qwen3-8b")
            col.plotly_chart(fig, width="stretch")
            col.caption(" · ".join(f"{C.pct(r.budget)} {r.gain:+.1f}" + ("*" if r.lo > 0 else "")
                                   for r in g.sort_values("budget").itertuples()) + "  (* significant)")

# ---------------------------------------------------------------------------------------------------------- 4
with tabs[3]:
    st.subheader("Long generations: 4K-token input, 128 → 16K output tokens")
    if dec.empty:
        st.info("No decode-scaling runs yet.")
    else:
        names = {"full": "Full cache", "ov": "OpenVINO eviction 10%", "ov+R": "Eviction 10% + RestoreKV"}
        colors = {"Full cache": C.FULL, "OpenVINO eviction 10%": C.EVICT, "Eviction 10% + RestoreKV": C.RKV}
        x = dec[(dec.batch == 1) & dec.status.eq("ok") & (dec.protocol == "aware")].copy()
        x["Configuration"], x["Model"] = x["config"].map(names), x["model"].map(C.MODELS)
        x["Output"] = x["output_tokens"].map(C.fmt_tokens)
        m = st.radio("Model", [v for v in C.MODELS.values() if v in set(x.Model)], horizontal=True)
        xm = x[x.Model == m].sort_values("output_tokens")
        c = st.columns(2)
        for col, y, t in ((c[0], "kv_mb_per_seq_end", "KV cache per sequence (MB)"),
                          (c[1], "tpot_ms", "Time per output token (ms)")):
            fig = px.line(xm, x="Output", y=y, color="Configuration", markers=True, color_discrete_map=colors,
                          category_orders={"Configuration": list(names.values())}, title=t,
                          labels={"Output": "generated output tokens", y: ""})
            fig.update_xaxes(type="category")
            fig.update_layout(height=380, hovermode="x unified")
            col.plotly_chart(fig, width="stretch")
        b8 = dec[(dec.batch == 8) & (dec.protocol == "aware")].copy()
        if len(b8):
            b8["Configuration"], b8["Model"] = b8["config"].map(names), b8["model"].map(C.MODELS)
            b8["cell"] = b8.apply(lambda r: f"{r.wall_tok_s:.1f}" if r.status == "ok" else "does not fit", axis=1)
            tb = b8.pivot_table(index=["Model", "output_tokens"], columns="Configuration", values="cell", aggfunc="first")
            tb = tb[[c_ for c_ in names.values() if c_ in tb.columns]]
            tb.index = [f"{a} · {C.fmt_tokens(o)} output" for a, o in tb.index]
            st.markdown("**Eight requests at once: total tokens/s**")
            st.dataframe(tb, width="stretch")
        C.note("Eviction caps the cache at 10% of the input (~400 tokens), so the KV cache stays at ~58 MB per sequence "
               "for Qwen3-8B however long the output gets, and the time per token stops growing. RestoreKV costs "
               "nothing here: same cache, same speed. Its one extra step is the restore pass, ~1.5% of the prefill "
               "time at 4K.")

# ---------------------------------------------------------------------------------------------------------- 5
with tabs[4]:
    ladder = load("ov_prototype", "*_ruler4k_ladder_n10")
    st.subheader("OpenVINO's cache layout is what limits the gain")
    st.markdown("The paper's eviction (KVzip) decides per head and per token; OpenVINO decides per 16-token block, the "
                "same for all heads. Each step towards OpenVINO's layout costs accuracy, and RestoreKV gains most when "
                "eviction is fine-grained (prototype, 130 samples, released adapter, 32-token blocks).")
    if len(ladder):
        b = st.radio("Budget", sorted(x for x in ladder.budget_pct.unique() if x < 100), index=1,
                     format_func=lambda v: f"{v:g}%", horizontal=True)
        g = ladder[(ladder.budget_pct == b) & ladder.base.isin(LADDER)]
        rows = [{"layout": LADDER[k_], "RestoreKV": r, "score": task_mean(x_)} for (k_, r), x_ in
                g.groupby(["base", "RestoreKV"])]
        t = pd.DataFrame(rows)
        t["Configuration"] = t["RestoreKV"].map({"without": "without RestoreKV", "with RestoreKV": "with RestoreKV"})
        fig = px.bar(t, x="layout", y="score", color="Configuration", barmode="group", text_auto=".1f",
                     category_orders={"layout": list(LADDER.values()),
                                      "Configuration": ["without RestoreKV", "with RestoreKV"]},
                     color_discrete_map={"without RestoreKV": C.EVICT, "with RestoreKV": C.RKV},
                     labels={"layout": "", "score": "RULER-4K score"}, range_y=[0, 100])
        fig.update_traces(textposition="outside")
        fig.update_layout(height=440)
        st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------------------------------------- 6
with tabs[5]:
    st.markdown("Individual answers: pick a run, a task and a sample.")
    if len(ruler):
        r_all = ruler.copy()
        r_all["Run"] = r_all["protocol"].map(lambda p: C.PROTOCOLS[p][0]) + " · " + r_all["model"].map(C.MODELS)
        c = st.columns(3)
        run = c[0].selectbox("Run", sorted(r_all.Run.unique()))
        r = r_all[r_all.Run == run]
        task = c[1].selectbox("Task", sorted(r.task.unique()))
        idx = c[2].selectbox("Sample", sorted(r[r.task == task].idx.unique(), key=str))
        s = r[(r.task == task) & (r.idx == idx)].sort_values(["budget", "variant"])
        raw = {}
        for name in s["run"].unique():
            for d in (C.STAGE4(), C.STAGE3()):
                f = d / name / "results.jsonl"
                if f.exists():
                    raw[name] = pd.DataFrame([json.loads(x_) for x_ in f.read_text().splitlines() if x_.strip()])
        det = pd.concat([v[v.idx == idx] for v in raw.values()], ignore_index=True) if raw else pd.DataFrame()
        if len(det):
            if "variant" not in det:
                det["variant"] = det["method"]
            st.markdown(f"Expected answer: `{det['answer'].iloc[0]}`")
            st.dataframe(det[["variant", "budget", "score", "pred"]].sort_values(["budget", "variant"]),
                         width="stretch", hide_index=True)

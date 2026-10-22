import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent))
import theme as T  # noqa: E402


# ============================================================
# PAGE CONFIG
# ============================================================

T.page("KV cache & memory scaling", "🧠")


# ============================================================
# HELPERS
# ============================================================

DATA_DIR = Path(__file__).resolve().parent / "data"


def pretty_name(name):
    replacements = {
        "kv_estimated_mb": "Estimated KV Cache (MB)",
        "kv_estimated_gb": "Estimated KV Cache (GB)",
        "max_new_tokens": "Requested Output Tokens",
        "requested_input_tokens": "Requested Input Tokens",
        "generated_tokens": "Generated Tokens",
        "total_sequence_tokens": "Total Sequence Tokens",
        "system_peak_ram_used_mb": "Peak System RAM Used (MB)",
        "process_peak_rss_mb": "Peak Process RSS (MB)",
        "process_rss_delta_mb": "Process RSS Delta (MB)",
        "process_disk_read_mb_generation": "Disk Read During Generation (MB)",
        "process_disk_write_mb_generation": "Disk Write During Generation (MB)",
        "system_pswpin_delta": "Swap In (pages)",
        "system_pswpout_delta": "Swap Out (pages)",
        "process_peak_swap_mb": "Peak Process Swap (MB)",
        "npu_allocated_mem_after_mb": "NPU Allocated Memory (MB)",
        "npu_allocated_mem_delta_mb": "NPU Allocated Memory Delta (MB)",
        "ttft_ms": "TTFT (ms)",
        "tpot_ms_per_token": "TPOT (ms/token)",
        "throughput_tokens_per_second": "Throughput (tokens/s)",
        "generate_duration_ms": "Generation Time (ms)",
        "inference_duration_ms": "Inference Time (ms)",
    }
    return replacements.get(name, name.replace("_", " ").title())


def fmt_tokens(value):
    if pd.isna(value):
        return "-"
    value = int(value)
    if value >= 1024 and value % 1024 == 0:
        return f"{value // 1024}K"
    return f"{value:,}"


def add_discrete_xaxis(fig, values):
    values = [v for v in values if pd.notna(v)]
    values = sorted(values)
    fig.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=[str(v) for v in values],
    )


def style_fig(fig, height=480):
    fig.update_layout(height=height, hovermode="x unified", legend_title_text="")
    return fig


def colors(column):
    """Fixed colour per model / device / status, the same under every filter."""
    return T.color_map(column, df[column].astype(str)) if column in df.columns else {}


# ============================================================
# FIND CSV
# ============================================================

csv_files = sorted(DATA_DIR.glob("*.csv"))

if not csv_files:
    st.error(
        "No CSV file found in the data/ directory. "
        "Place the benchmark CSV inside data/ and reload."
    )
    st.stop()

default_csv = next(
    (
        f for f in csv_files
        if "experiment2" in f.name.lower()
        or "kv_cache" in f.name.lower()
    ),
    csv_files[0],
)

selected_file = st.sidebar.selectbox(
    "Results file",
    csv_files,
    index=csv_files.index(default_csv),
    format_func=lambda p: p.name,
)

df = pd.read_csv(selected_file)


# ============================================================
# NORMALIZE IMPORTANT COLUMNS
# ============================================================

numeric_candidates = [
    "requested_input_tokens",
    "synthetic_prompt_tokens",
    "input_tokens",
    "max_new_tokens",
    "generated_tokens",
    "total_sequence_tokens",
    "load_time_ms",
    "ttft_ms",
    "tpot_ms_per_token",
    "generate_duration_ms",
    "inference_duration_ms",
    "throughput_tokens_per_second",
    "process_peak_rss_mb",
    "process_rss_delta_mb",
    "process_peak_swap_mb",
    "system_peak_ram_used_mb",
    "system_peak_swap_used_mb",
    "process_disk_read_mb_generation",
    "process_disk_write_mb_generation",
    "system_pswpin_delta",
    "system_pswpout_delta",
    "system_pgmajfault_delta",
    "npu_allocated_mem_before_mb",
    "npu_allocated_mem_after_mb",
    "npu_allocated_mem_delta_mb",
    "kv_layers",
    "kv_heads",
    "kv_head_dim",
    "kv_bytes_per_element",
    "kv_estimated_mb",
    "kv_estimated_gb",
    "peak_sequence_tokens",
]

for col in numeric_candidates:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


# ============================================================
# HEADER
# ============================================================

T.hero(
    "Long-generation stress test",
    "KV cache & memory scaling",
    "A fixed input context with a growing generation length: how the KV cache grows, what it does to RAM and "
    "device memory, disk and swap activity, and where each model and device stops succeeding.",
    ["models: " + ", ".join(sorted(df["model"].dropna().unique())) if "model" in df.columns else "",
     "devices: " + ", ".join(sorted(df["device"].dropna().unique())) if "device" in df.columns else "",
     f"{len(df)} runs"],
)

T.note(
    "<b>How to read the memory numbers.</b> <code>kv_estimated_*</code> is an estimate of the KV-cache footprint "
    "computed from the model's cache shape (layers × KV heads × head size × bytes × tokens), not a measured "
    "hardware allocation. Disk reads and writes are the process's observed I/O during generation; on their own they "
    "do not show that the KV cache was offloaded to the SSD."
)


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.markdown("### Filters")

# The NPU sweep was re-run with a corrected prompt-length setting: the CSV keeps both attempts. By default only the
# latest attempt of every configuration counts, so superseded failures do not show up as limits.
config_key = [c for c in ("model", "device", "max_new_tokens", "requested_input_tokens") if c in df.columns]
if "timestamp" in df.columns and config_key:
    n_all = len(df)
    latest_only = st.sidebar.toggle(
        "Latest attempt per configuration",
        value=True,
        help="Re-runs replace earlier attempts of the same model, device and length "
             "(the first NPU sweep failed on a prompt-length setting and was repeated).",
    )
    if latest_only:
        df = (df.assign(_ts=pd.to_datetime(df["timestamp"], errors="coerce"))
                .sort_values("_ts").drop_duplicates(config_key, keep="last").drop(columns="_ts"))
        st.sidebar.caption(f"{len(df)} of {n_all} runs (latest attempts)")

models = sorted(df["model"].dropna().unique().tolist()) if "model" in df.columns else []
devices = sorted(df["device"].dropna().unique().tolist()) if "device" in df.columns else []

default_models = models
default_devices = devices

selected_models = st.sidebar.multiselect(
    "Model",
    models,
    default=default_models,
)

selected_devices = st.sidebar.multiselect(
    "Device",
    devices,
    default=default_devices,
)

if "status" in df.columns:
    status_options = sorted(df["status"].dropna().unique().tolist())
    selected_status = st.sidebar.multiselect(
        "Run Status",
        status_options,
        default=status_options,
    )
else:
    selected_status = []

if "max_new_tokens" in df.columns:
    output_values = sorted(df["max_new_tokens"].dropna().unique().tolist())
    selected_outputs = st.sidebar.multiselect(
        "Requested Output Tokens",
        output_values,
        default=output_values,
        format_func=fmt_tokens,
    )
else:
    selected_outputs = []


# ============================================================
# FILTER
# ============================================================

filtered = df.copy()

if selected_models:
    filtered = filtered[filtered["model"].isin(selected_models)]

if selected_devices:
    filtered = filtered[filtered["device"].isin(selected_devices)]

if selected_status and "status" in filtered.columns:
    filtered = filtered[filtered["status"].isin(selected_status)]

if selected_outputs and "max_new_tokens" in filtered.columns:
    filtered = filtered[
        filtered["max_new_tokens"].isin(selected_outputs)
    ]

success = filtered[
    filtered["status"].eq("SUCCESS")
].copy() if "status" in filtered.columns else filtered.copy()

failed = filtered[
    ~filtered["status"].eq("SUCCESS")
].copy() if "status" in filtered.columns else pd.DataFrame()


# ============================================================
# OVERVIEW
# ============================================================

st.subheader("Overview")

c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:
    st.metric("Runs", len(filtered))

with c2:
    st.metric("Successful", len(success))

with c3:
    st.metric("Failed", len(failed))

with c4:
    st.metric(
        "Max Generated",
        fmt_tokens(success["generated_tokens"].max())
        if not success.empty and "generated_tokens" in success.columns
        else "-",
    )

with c5:
    st.metric(
        "Max Estimated KV",
        f"{success['kv_estimated_gb'].max():.2f} GB"
        if not success.empty and "kv_estimated_gb" in success.columns
        else "-",
    )

with c6:
    st.metric(
        "Max Peak RAM",
        f"{success['system_peak_ram_used_mb'].max()/1024:.2f} GB"
        if not success.empty and "system_peak_ram_used_mb" in success.columns
        else "-",
    )

# ============================================================
# TABS
# ============================================================

tab_kv, tab_ram, tab_perf, tab_storage, tab_limits = st.tabs(
    [
        "KV cache growth",
        "RAM & device memory",
        "Generation performance",
        "Disk & swap",
        "Limits & failures",
    ]
)


# ============================================================
# TAB 1 — KV CACHE
# ============================================================

with tab_kv:

    st.subheader("How does KV Cache grow with generation length?")

    if success.empty:
        st.warning("No successful runs match the current filters.")
    else:

        col1, col2, col3 = st.columns(3)

        with col1:
            kv_metric = st.selectbox(
                "KV metric",
                ["kv_estimated_gb", "kv_estimated_mb"],
                format_func=pretty_name,
                key="kv_metric",
            )

        with col2:
            kv_compare = st.selectbox(
                "Compare by",
                ["model", "device"],
                format_func=pretty_name,
                key="kv_compare",
            )

        with col3:
            kv_chart = st.selectbox(
                "Chart",
                ["Line Chart", "Bar Chart"],
                key="kv_chart",
            )

        plot = (
            success
            .groupby(
                ["max_new_tokens", kv_compare],
                dropna=False
            )[kv_metric]
            .mean()
            .reset_index()
        )

        if kv_chart == "Line Chart":
            fig = px.line(
                plot,
                x="max_new_tokens",
                y=kv_metric,
                color=kv_compare,
                color_discrete_map=colors(kv_compare),
                markers=True,
                title=f"{pretty_name(kv_metric)} vs Generated Tokens",
            )
        else:
            fig = px.bar(
                plot,
                x="max_new_tokens",
                y=kv_metric,
                color=kv_compare,
                color_discrete_map=colors(kv_compare),
                barmode="group",
                title=f"{pretty_name(kv_metric)} vs Generated Tokens",
            )

        style_fig(fig)
        add_discrete_xaxis(
            fig,
            sorted(success["max_new_tokens"].dropna().unique())
        )
        fig.update_layout(
            xaxis_title="Requested Output Tokens",
            yaxis_title=pretty_name(kv_metric),
        )
        st.plotly_chart(fig, width="stretch")

        T.note(
            "With the input held at a fixed prompt length, the estimated KV-cache footprint grows linearly with "
            "every generated token (the x axis doubles at each step, so the curve bends upward; the plot against "
            "total sequence length below is a straight line). Qwen3-4B and Qwen3-8B have the same KV-cache shape "
            "(36 layers × 8 KV heads × 128), so their lines lie on top of each other."
        )

        st.subheader("KV Cache vs Total Sequence Length")

        seq = (
            success
            .groupby(
                ["total_sequence_tokens", "model"],
                dropna=False
            )["kv_estimated_gb"]
            .mean()
            .reset_index()
        )

        fig2 = px.line(
            seq,
            x="total_sequence_tokens",
            y="kv_estimated_gb",
            color="model",
            color_discrete_map=colors("model"),
            markers=True,
            title="Estimated KV Cache vs Total Sequence Length",
        )
        style_fig(fig2)
        fig2.update_layout(
            xaxis_title="Total Sequence Tokens",
            yaxis_title="Estimated KV Cache (GB)",
        )
        st.plotly_chart(fig2, width="stretch")

        with st.expander("KV Cache Data"):
            st.dataframe(
                success[
                    [
                        c for c in [
                            "model",
                            "device",
                            "max_new_tokens",
                            "generated_tokens",
                            "total_sequence_tokens",
                            "kv_layers",
                            "kv_heads",
                            "kv_head_dim",
                            "kv_bytes_per_element",
                            "kv_estimated_mb",
                            "kv_estimated_gb",
                        ]
                        if c in success.columns
                    ]
                ].sort_values(
                    ["model", "max_new_tokens"]
                ),
                width="stretch",
            )


# ============================================================
# TAB 2 — RAM / DEVICE MEMORY
# ============================================================

with tab_ram:

    st.subheader("How does KV growth translate into memory pressure?")

    if success.empty:
        st.warning("No successful runs match the current filters.")
    else:

        memory_metric_options = [
            c for c in [
                "system_peak_ram_used_mb",
                "process_peak_rss_mb",
                "process_rss_delta_mb",
                "npu_allocated_mem_after_mb",
                "npu_allocated_mem_delta_mb",
            ]
            if c in success.columns
        ]

        mcol1, mcol2, mcol3 = st.columns(3)

        with mcol1:
            memory_metric = st.selectbox(
                "Memory metric",
                memory_metric_options,
                format_func=pretty_name,
                key="memory_metric",
            )

        with mcol2:
            memory_compare = st.selectbox(
                "Compare by",
                ["model", "device"],
                format_func=pretty_name,
                key="memory_compare",
            )

        with mcol3:
            memory_chart = st.selectbox(
                "Chart",
                ["Line Chart", "Bar Chart"],
                key="memory_chart",
            )

        mem_plot = (
            success
            .groupby(
                ["max_new_tokens", memory_compare],
                dropna=False
            )[memory_metric]
            .mean()
            .reset_index()
        )

        if memory_chart == "Line Chart":
            fig = px.line(
                mem_plot,
                x="max_new_tokens",
                y=memory_metric,
                color=memory_compare,
                color_discrete_map=colors(memory_compare),
                markers=True,
                title=f"{pretty_name(memory_metric)} vs Generated Tokens",
            )
        else:
            fig = px.bar(
                mem_plot,
                x="max_new_tokens",
                y=memory_metric,
                color=memory_compare,
                color_discrete_map=colors(memory_compare),
                barmode="group",
                title=f"{pretty_name(memory_metric)} vs Generated Tokens",
            )

        style_fig(fig)
        add_discrete_xaxis(
            fig,
            sorted(success["max_new_tokens"].dropna().unique())
        )
        fig.update_layout(
            xaxis_title="Requested Output Tokens",
            yaxis_title=pretty_name(memory_metric),
        )
        st.plotly_chart(fig, width="stretch")

        st.subheader("Estimated KV Cache vs Peak System RAM")

        ram_plot = success.dropna(
            subset=["kv_estimated_gb", "system_peak_ram_used_mb"]
        )

        fig2 = px.scatter(
            ram_plot,
            x="kv_estimated_gb",
            y="system_peak_ram_used_mb",
            color="model",
            color_discrete_map=colors("model"),
            symbol="device",
            hover_data=[
                "max_new_tokens",
                "generated_tokens",
            ],
            title="Estimated KV Cache vs Peak System RAM Used",
        )
        style_fig(fig2)
        fig2.update_layout(
            xaxis_title="Estimated KV Cache (GB)",
            yaxis_title="Peak System RAM Used (MB)",
        )
        st.plotly_chart(fig2, width="stretch")

        T.note(
            "For NPU runs, <code>npu_allocated_mem_after_mb</code> is a direct hardware-side signal: the benchmark "
            "records the NPU allocation itself. For CPU and GPU runs, process RSS and system RAM are not "
            "GPU-memory measurements."
        )


# ============================================================
# TAB 3 — GENERATION PERFORMANCE
# ============================================================

with tab_perf:

    st.subheader("How does longer generation affect decode performance?")

    if success.empty:
        st.warning("No successful runs match the current filters.")
    else:

        pcol1, pcol2, pcol3 = st.columns(3)

        performance_metrics = [
            c for c in [
                "tpot_ms_per_token",
                "throughput_tokens_per_second",
                "generate_duration_ms",
                "inference_duration_ms",
                "ttft_ms",
            ]
            if c in success.columns
        ]

        with pcol1:
            perf_metric = st.selectbox(
                "Performance metric",
                performance_metrics,
                format_func=pretty_name,
                key="perf_metric",
            )

        with pcol2:
            perf_compare = st.selectbox(
                "Compare by",
                ["model", "device"],
                format_func=pretty_name,
                key="perf_compare",
            )

        with pcol3:
            perf_chart = st.selectbox(
                "Chart",
                ["Line Chart", "Bar Chart"],
                key="perf_chart",
            )

        perf_plot = (
            success
            .groupby(
                ["max_new_tokens", perf_compare],
                dropna=False
            )[perf_metric]
            .mean()
            .reset_index()
        )

        if perf_chart == "Line Chart":
            fig = px.line(
                perf_plot,
                x="max_new_tokens",
                y=perf_metric,
                color=perf_compare,
                color_discrete_map=colors(perf_compare),
                markers=True,
                title=f"{pretty_name(perf_metric)} vs Generation Length",
            )
        else:
            fig = px.bar(
                perf_plot,
                x="max_new_tokens",
                y=perf_metric,
                color=perf_compare,
                color_discrete_map=colors(perf_compare),
                barmode="group",
                title=f"{pretty_name(perf_metric)} vs Generation Length",
            )

        style_fig(fig)
        add_discrete_xaxis(
            fig,
            sorted(success["max_new_tokens"].dropna().unique())
        )
        fig.update_layout(
            xaxis_title="Requested Output Tokens",
            yaxis_title=pretty_name(perf_metric),
        )
        st.plotly_chart(fig, width="stretch")

        st.subheader("Requested vs Actually Generated")

        generated = (
            success
            .groupby(
                ["max_new_tokens", "model"],
                dropna=False
            )["generated_tokens"]
            .mean()
            .reset_index()
        )

        fig2 = px.line(
            generated,
            x="max_new_tokens",
            y="generated_tokens",
            color="model",
            color_discrete_map=colors("model"),
            markers=True,
            title="Requested Output vs Actual Generated Tokens",
        )

        # Ideal line
        xvals = sorted(success["max_new_tokens"].dropna().unique())
        fig2.add_trace(
            go.Scatter(
                x=xvals,
                y=xvals,
                mode="lines",
                name="Requested = Generated",
                line=dict(dash="dash", color=T.FULL),
            )
        )

        style_fig(fig2)
        add_discrete_xaxis(fig2, xvals)
        fig2.update_layout(
            xaxis_title="Requested Output Tokens",
            yaxis_title="Actual Generated Tokens",
        )
        st.plotly_chart(fig2, width="stretch")

        st.caption(
            "Points below the dashed line indicate runs that did not "
            "generate the full requested sequence."
        )


# ============================================================
# TAB 4 — DISK / SWAP
# ============================================================

with tab_storage:

    st.subheader("Is long generation causing disk or swap activity?")

    if success.empty:
        st.warning("No successful runs match the current filters.")
    else:

        disk_cols = [
            c for c in [
                "process_disk_read_mb_generation",
                "process_disk_write_mb_generation",
                "process_peak_swap_mb",
                "system_peak_swap_used_mb",
                "system_pswpin_delta",
                "system_pswpout_delta",
            ]
            if c in success.columns
        ]

        if disk_cols:

            storage_metric = st.selectbox(
                "Storage / swap metric",
                disk_cols,
                format_func=pretty_name,
                key="storage_metric",
            )

            storage_compare = st.selectbox(
                "Compare by",
                ["model", "device"],
                format_func=pretty_name,
                key="storage_compare",
            )

            storage_plot = (
                success
                .groupby(
                    ["max_new_tokens", storage_compare],
                    dropna=False
                )[storage_metric]
                .mean()
                .reset_index()
            )

            fig = px.line(
                storage_plot,
                x="max_new_tokens",
                y=storage_metric,
                color=storage_compare,
                color_discrete_map=colors(storage_compare),
                markers=True,
                title=f"{pretty_name(storage_metric)} vs Generation Length",
            )

            style_fig(fig)
            add_discrete_xaxis(
                fig,
                sorted(success["max_new_tokens"].dropna().unique())
            )
            fig.update_layout(
                xaxis_title="Requested Output Tokens",
                yaxis_title=pretty_name(storage_metric),
            )
            st.plotly_chart(fig, width="stretch")

        # Summary cards
        s1, s2, s3, s4 = st.columns(4)

        with s1:
            st.metric(
                "Max Disk Read",
                f"{success['process_disk_read_mb_generation'].max():,.1f} MB"
                if "process_disk_read_mb_generation" in success.columns
                else "-"
            )

        with s2:
            st.metric(
                "Max Disk Write",
                f"{success['process_disk_write_mb_generation'].max():,.1f} MB"
                if "process_disk_write_mb_generation" in success.columns
                else "-"
            )

        with s3:
            st.metric(
                "Max Process Swap",
                f"{success['process_peak_swap_mb'].max():,.1f} MB"
                if "process_peak_swap_mb" in success.columns
                else "-"
            )

        with s4:
            st.metric(
                "Max Swap Out",
                f"{success['system_pswpout_delta'].max():,.0f}"
                if "system_pswpout_delta" in success.columns
                else "-"
            )

        T.note(
            "Disk reads and writes during generation are an I/O observation, not an attribution to the KV cache. "
            "Showing KV-cache offloading to the SSD would need an explicit cache / offload signal from the runtime."
        )


# ============================================================
# TAB 5 — LIMITS / FAILURES
# ============================================================

with tab_limits:

    st.subheader("Where does the workload stop succeeding?")

    if "status" in filtered.columns:

        # Success/failure matrix
        matrix = (
            filtered
            .assign(
                outcome=filtered["status"].map(
                    lambda x: "SUCCESS" if x == "SUCCESS" else "FAILED"
                )
            )
            .groupby(
                ["model", "device", "max_new_tokens"],
                dropna=False
            )["outcome"]
            .first()
            .reset_index()
        )

        matrix["model · device"] = matrix["model"].astype(str) + " · " + matrix["device"].astype(str)
        matrix["result"] = matrix["outcome"].map(
            {
                "SUCCESS": "✓ Success",
                "FAILED": "✗ Failed",
            }
        )

        fig = px.scatter(
            matrix,
            x="max_new_tokens",
            y="model · device",
            color="outcome",
            symbol="outcome",
            color_discrete_map={"SUCCESS": T.STATUS_COLORS["SUCCESS"], "FAILED": T.STATUS_COLORS["FAILED"]},
            symbol_map={"SUCCESS": "circle", "FAILED": "x"},
            hover_data=["result"],
            title="Success / failure frontier (one row per model and device)",
        )
        style_fig(fig)
        add_discrete_xaxis(
            fig,
            sorted(matrix["max_new_tokens"].dropna().unique())
        )
        fig.update_layout(
            xaxis_title="Requested Output Tokens",
            yaxis_title="",
        )
        fig.update_traces(marker=dict(size=11))
        st.plotly_chart(fig, width="stretch")

        # Failure table
        if not failed.empty:

            st.subheader("Failed Runs")

            failure_cols = [
                c for c in [
                    "model",
                    "device",
                    "max_new_tokens",
                    "failure_stage",
                    "error_type",
                    "error_message",
                ]
                if c in failed.columns
            ]

            st.dataframe(
                failed[failure_cols].sort_values(
                    ["model", "device", "max_new_tokens"]
                ),
                width="stretch",
            )

            st.subheader("Failure Reasons")

            if "error_message" in failed.columns:

                failure_counts = (
                    failed
                    .assign(
                        reason=failed["error_message"]
                        .fillna("Unknown")
                        .str.slice(0, 120)
                    )
                    .groupby("reason")
                    .size()
                    .reset_index(name="runs")
                    .sort_values("runs", ascending=False)
                )

                st.dataframe(
                    failure_counts,
                    width="stretch",
                )

        else:
            st.success("No failed runs match the current filters.")


# ============================================================
# RAW DATA
# ============================================================

st.divider()

with st.expander("Filtered runs (raw data)"):

    st.dataframe(
        filtered,
        width="stretch",
    )

st.download_button(
    label="Download filtered runs (CSV)",
    data=filtered.to_csv(index=False),
    file_name="kv_cache_scaling_filtered.csv",
    mime="text/csv",
)

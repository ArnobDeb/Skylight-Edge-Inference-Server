import streamlit as st
import pandas as pd
import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="OpenVINO Benchmark Dashboard",
    page_icon="⚡",
    layout="wide"
)


# ============================================================
# SESSION STATE HELPERS
# ============================================================

def initialize_state(key, value):
    """
    Initialize a session-state value only once.

    Streamlit reruns the entire script whenever a widget changes.
    Using session_state prevents previously selected values from
    unnecessarily returning to their defaults.
    """
    if key not in st.session_state:
        st.session_state[key] = value


def validate_state(key, valid_options, fallback):
    """
    Keep an existing selection if it is still valid.

    If the available options changed and the previous selection
    is no longer valid, use the supplied fallback.
    """
    if key not in st.session_state:
        st.session_state[key] = fallback

    elif st.session_state[key] not in valid_options:
        st.session_state[key] = fallback


# ============================================================
# LOAD DATA
# ============================================================

CSV_PATH = "data/old_perf.csv"

df = pd.read_csv(CSV_PATH)


# ============================================================
# HEADER
# ============================================================

st.title("⚡ OpenVINO Benchmark Dashboard")

st.caption(
    "LLM inference performance profiling on Intel AI PCs"
)

st.divider()


# ============================================================
# IDENTIFY COLUMN TYPES
# ============================================================

numeric_columns = df.select_dtypes(
    include="number"
).columns.tolist()

categorical_columns = df.select_dtypes(
    exclude="number"
).columns.tolist()


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header("Experiment Filters")


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

if "model" in df.columns:

    models = df["model"].dropna().unique().tolist()

    initialize_state(
        "selected_models",
        models
    )

    selected_models = st.sidebar.multiselect(
        "Model",
        models,
        key="selected_models"
    )

else:

    selected_models = []


# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

if "device" in df.columns:

    devices = df["device"].dropna().unique().tolist()

    initialize_state(
        "selected_devices",
        devices
    )

    selected_devices = st.sidebar.multiselect(
        "Device",
        devices,
        key="selected_devices"
    )

else:

    selected_devices = []


# ------------------------------------------------------------
# Input Tokens
# ------------------------------------------------------------

if "requested_input_tokens" in df.columns:

    input_tokens = sorted(
        df["requested_input_tokens"]
        .dropna()
        .unique()
        .tolist()
    )

    initialize_state(
        "selected_input_tokens",
        input_tokens
    )

    selected_input_tokens = st.sidebar.multiselect(
        "Requested Input Tokens",
        input_tokens,
        key="selected_input_tokens"
    )

else:

    selected_input_tokens = []


# ------------------------------------------------------------
# Output Tokens
# ------------------------------------------------------------

if "max_new_tokens" in df.columns:

    output_tokens = sorted(
        df["max_new_tokens"]
        .dropna()
        .unique()
        .tolist()
    )

    initialize_state(
        "selected_output_tokens",
        output_tokens
    )

    selected_output_tokens = st.sidebar.multiselect(
        "Max New Tokens",
        output_tokens,
        key="selected_output_tokens"
    )

else:

    selected_output_tokens = []


# ============================================================
# FILTER DATA
# ============================================================

filtered_df = df.copy()


# ------------------------------------------------------------
# Model filter
# ------------------------------------------------------------

if selected_models:

    filtered_df = filtered_df[
        filtered_df["model"].isin(selected_models)
    ]


# ------------------------------------------------------------
# Device filter
# ------------------------------------------------------------

if selected_devices:

    filtered_df = filtered_df[
        filtered_df["device"].isin(selected_devices)
    ]


# ------------------------------------------------------------
# Input token filter
# ------------------------------------------------------------

if selected_input_tokens:

    filtered_df = filtered_df[
        filtered_df["requested_input_tokens"].isin(
            selected_input_tokens
        )
    ]


# ------------------------------------------------------------
# Output token filter
# ------------------------------------------------------------

if selected_output_tokens:

    filtered_df = filtered_df[
        filtered_df["max_new_tokens"].isin(
            selected_output_tokens
        )
    ]


# ============================================================
# DATASET SUMMARY
# ============================================================

st.subheader("Benchmark Overview")

summary_col1, summary_col2, summary_col3, summary_col4 = (
    st.columns(4)
)


with summary_col1:

    st.metric(
        "Experiments",
        len(filtered_df)
    )


with summary_col2:

    st.metric(
        "Models",
        filtered_df["model"].nunique()
        if "model" in filtered_df.columns
        else "-"
    )


with summary_col3:

    st.metric(
        "Devices",
        filtered_df["device"].nunique()
        if "device" in filtered_df.columns
        else "-"
    )


with summary_col4:

    st.metric(
        "Input Configurations",
        filtered_df["requested_input_tokens"].nunique()
        if "requested_input_tokens" in filtered_df.columns
        else "-"
    )


st.divider()


# ============================================================
# PERFORMANCE VISUALIZATION
# ============================================================

st.subheader("Performance Visualization")


# ============================================================
# GRAPH CONTROLS
# ============================================================

control_col1, control_col2, control_col3, control_col4, control_col5 = (
    st.columns(5)
)


# ------------------------------------------------------------
# Metric
# ------------------------------------------------------------

with control_col1:

    if numeric_columns:

        initialize_state(
            "metric",
            numeric_columns[0]
        )

        validate_state(
            "metric",
            numeric_columns,
            numeric_columns[0]
        )

        metric = st.selectbox(
            "Metric",
            numeric_columns,
            key="metric",
            format_func=lambda x: x.replace("_", " ").title()
        )

    else:

        st.error("No numeric metrics were found in the CSV.")

        st.stop()


# ------------------------------------------------------------
# X-axis
# ------------------------------------------------------------

with control_col2:

    x_axis_options = [
        column
        for column in categorical_columns + numeric_columns
        if column != metric
    ]

    if not x_axis_options:

        st.error(
            "No valid X-axis columns are available."
        )

        st.stop()

    initialize_state(
        "x_axis",
        x_axis_options[0]
    )

    validate_state(
        "x_axis",
        x_axis_options,
        x_axis_options[0]
    )

    x_axis = st.selectbox(
        "X-axis",
        x_axis_options,
        key="x_axis",
        format_func=lambda x: x.replace("_", " ").title()
    )


# ------------------------------------------------------------
# Group By
# ------------------------------------------------------------

with control_col3:

    group_options = [
        "None"
    ] + [
        column
        for column in categorical_columns + numeric_columns
        if column != metric and column != x_axis
    ]

    initialize_state(
        "group_by",
        "None"
    )

    validate_state(
        "group_by",
        group_options,
        "None"
    )

    group_by = st.selectbox(
        "Group by",
        group_options,
        key="group_by",
        format_func=lambda x: x.replace("_", " ").title()
    )


# ------------------------------------------------------------
# Aggregation
# ------------------------------------------------------------

with control_col4:

    aggregation_options = [
        "Mean",
        "Median",
        "Minimum",
        "Maximum"
    ]

    initialize_state(
        "aggregation",
        "Mean"
    )

    aggregation = st.selectbox(
        "Aggregation",
        aggregation_options,
        key="aggregation"
    )


# ------------------------------------------------------------
# Compare By
# ------------------------------------------------------------

with control_col5:

    compare_options = [
        "None"
    ] + [
        column
        for column in categorical_columns + numeric_columns
        if column != metric and column != x_axis
    ]

    initialize_state(
        "compare_by",
        "None"
    )

    validate_state(
        "compare_by",
        compare_options,
        "None"
    )

    compare_by = st.selectbox(
        "Compare By",
        compare_options,
        key="compare_by",
        format_func=lambda x: x.replace("_", " ").title()
    )


# ============================================================
# CHART TYPE
# ============================================================

chart_types = [
    "Bar Chart",
    "Line Chart",
    "Scatter Plot"
]

initialize_state(
    "chart_type",
    "Bar Chart"
)

chart_type = st.radio(
    "Chart Type",
    chart_types,
    key="chart_type",
    horizontal=True
)


# ============================================================
# SELECTED METRIC STATISTICS
# ============================================================

if not filtered_df.empty:

    metric_values = filtered_df[metric].dropna()

    if len(metric_values) > 0:

        st.subheader("Selected Metric")

        stat_col1, stat_col2, stat_col3, stat_col4 = (
            st.columns(4)
        )

        with stat_col1:

            st.metric(
                "Mean",
                f"{metric_values.mean():,.2f}"
            )

        with stat_col2:

            st.metric(
                "Median",
                f"{metric_values.median():,.2f}"
            )

        with stat_col3:

            st.metric(
                "Minimum",
                f"{metric_values.min():,.2f}"
            )

        with stat_col4:

            st.metric(
                "Maximum",
                f"{metric_values.max():,.2f}"
            )


st.divider()


# ============================================================
# AGGREGATE DATA
# ============================================================

if filtered_df.empty:

    st.warning(
        "No experiments match the selected filters."
    )

else:

    # --------------------------------------------------------
    # Determine aggregation function
    # --------------------------------------------------------

    aggregation_functions = {
        "Mean": "mean",
        "Median": "median",
        "Minimum": "min",
        "Maximum": "max"
    }

    agg_function = aggregation_functions[
        aggregation
    ]


    # --------------------------------------------------------
    # Grouping columns
    # --------------------------------------------------------

    grouping_columns = [
        x_axis
    ]

    if group_by != "None":

        grouping_columns.append(
            group_by
        )

    if (
        compare_by != "None"
        and compare_by not in grouping_columns
    ):

        grouping_columns.append(
            compare_by
        )


    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    plot_df = (
        filtered_df
        .groupby(
            grouping_columns,
            dropna=False
        )[metric]
        .agg(agg_function)
        .reset_index()
    )


    # ========================================================
    # DETERMINE COLOR COLUMN
    # ========================================================

    if compare_by != "None":

        color_column = compare_by

    elif group_by != "None":

        color_column = group_by

    else:

        color_column = None


    # ========================================================
    # X-AXIS DISPLAY
    # ========================================================

    # Numeric benchmark configurations such as:
    #
    # 128, 256, 512, 1024, 2048, 4096, 8192
    #
    # should be displayed as discrete experimental points
    # instead of being treated as a continuous numerical axis.

    if x_axis in numeric_columns:

        x_values = sorted(
            plot_df[x_axis]
            .dropna()
            .unique()
        )

        plot_df["_x_display"] = (
            plot_df[x_axis]
            .astype(str)
        )

        plot_x = "_x_display"

    else:

        plot_x = x_axis

        x_values = None


    # ========================================================
    # GRAPH TITLE
    # ========================================================

    graph_title = (
        f"{metric.replace('_', ' ').title()} "
        f"vs "
        f"{x_axis.replace('_', ' ').title()}"
    )


    if compare_by != "None":

        graph_title += (
            f" — compared by "
            f"{compare_by.replace('_', ' ').title()}"
        )


    if group_by != "None":

        graph_title += (
            f" — grouped by "
            f"{group_by.replace('_', ' ').title()}"
        )


    # ========================================================
    # CREATE GRAPH
    # ========================================================

    if chart_type == "Bar Chart":

        fig = px.bar(
            plot_df,
            x=plot_x,
            y=metric,
            color=color_column,
            barmode="group",
            title=graph_title
        )


    elif chart_type == "Line Chart":

        fig = px.line(
            plot_df,
            x=plot_x,
            y=metric,
            color=color_column,
            markers=True,
            title=graph_title
        )


    else:

        fig = px.scatter(
            plot_df,
            x=plot_x,
            y=metric,
            color=color_column,
            title=graph_title
        )


    # ========================================================
    # GRAPH LAYOUT
    # ========================================================

    fig.update_layout(

        height=550,

        hovermode="x unified",

        margin=dict(
            l=40,
            r=40,
            t=80,
            b=60
        ),

        title_x=0.02,

        title_font=dict(
            size=20
        ),

        legend_title_text=(
            compare_by.replace("_", " ").title()
            if compare_by != "None"
            else (
                group_by.replace("_", " ").title()
                if group_by != "None"
                else ""
            )
        ),

        xaxis_title=(
            x_axis.replace("_", " ").title()
        ),

        yaxis_title=(
            metric.replace("_", " ").title()
        )
    )


    # ========================================================
    # DISCRETE X-AXIS
    # ========================================================

    if x_values is not None:

        fig.update_xaxes(

            type="category",

            categoryorder="array",

            categoryarray=[
                str(value)
                for value in x_values
            ]
        )


    # ========================================================
    # DISPLAY GRAPH
    # ========================================================

    st.plotly_chart(
        fig,
        # use_container_width=True
        width='stretch'
    )


    # ========================================================
    # AGGREGATED DATA
    # ========================================================

    with st.expander(
        "View Aggregated Data"
    ):

        st.dataframe(
            plot_df,
            # use_container_width=True
            width='stretch'
        )


# ============================================================
# DOWNLOAD FILTERED RESULTS
# ============================================================

st.download_button(

    label="⬇ Download Filtered Results",

    data=filtered_df.to_csv(
        index=False
    ),

    file_name="filtered_benchmark_results.csv",

    mime="text/csv"
)


# ============================================================
# RAW DATA
# ============================================================

with st.expander(
    "View Filtered Benchmark Data"
):

    st.dataframe(
        filtered_df,
        # use_container_width=True
        width='stretch'
    )
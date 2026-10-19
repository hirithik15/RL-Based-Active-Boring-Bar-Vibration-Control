
"""Streamlit dashboard for simulated active boring-bar vibration control.

The dashboard communicates with api.py. It does not control physical machinery.
"""

import requests
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


st.set_page_config(
    page_title="Active Boring-Bar Vibration Control",
    page_icon="📉",
    layout="wide",
)

API_BASE = st.sidebar.text_input(
    "API URL",
    value="http://127.0.0.1:8000",
).strip().rstrip("/")

CONTROLLERS = ["No control", "PID", "SAC"]
COLORS = {
    "No control": "#7f8c8d",
    "PID": "#e67e22",
    "SAC": "#2471a3",
}


def api_get(path):
    response = requests.get(f"{API_BASE}{path}", timeout=10)
    response.raise_for_status()
    return response.json()


def api_post(path, payload):
    response = requests.post(
        f"{API_BASE}{path}",
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def metric_value(metrics, key):
    value = metrics.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def make_comparison_table(results):
    rows = []

    for result in results:
        rows.append({
            "Controller": result["controller"],
            "Steady RMS (µm)": metric_value(
                result, "steady_rms_vibration_um"
            ),
            "Overall RMS (µm)": metric_value(
                result, "rms_vibration_um"
            ),
            "Peak vibration (µm)": metric_value(
                result, "peak_vibration_um"
            ),
            "RMS force (N)": metric_value(
                result, "rms_control_force_n"
            ),
            "Peak force (N)": metric_value(
                result, "peak_control_force_n"
            ),
            "Saturation (%)": 100 * metric_value(
                result, "actuator_saturation_fraction"
            ),
        })

    return pd.DataFrame(rows)


def make_trace_plot(traces, y_column, title, y_label):
    fig = go.Figure()

    for controller in CONTROLLERS:
        if controller not in traces:
            continue

        data = traces[controller]
        fig.add_trace(
            go.Scatter(
                x=data["time_s"],
                y=data[y_column],
                name=controller,
                mode="lines",
                line={"color": COLORS[controller], "width": 1.8},
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title=y_label,
        hovermode="x unified",
        legend_title="Controller",
        margin={"l": 15, "r": 15, "t": 60, "b": 15},
    )
    return fig


def make_force_disturbance_plot(traces):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=(
            "Applied disturbance force",
            "Controller force",
        ),
    )

    for controller in CONTROLLERS:
        if controller not in traces:
            continue

        data = traces[controller]
        color = COLORS[controller]

        fig.add_trace(
            go.Scatter(
                x=data["time_s"],
                y=data["disturbance_force_n"],
                name=f"{controller} disturbance",
                line={"color": color, "width": 1.5},
                legendgroup=controller,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=data["time_s"],
                y=data["control_force_n"],
                name=f"{controller} control",
                line={"color": color, "width": 1.5},
                legendgroup=controller,
            ),
            row=2,
            col=1,
        )

    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Force (N)", row=1, col=1)
    fig.update_yaxes(title_text="Force (N)", row=2, col=1)
    fig.update_layout(
        height=650,
        hovermode="x unified",
        margin={"l": 15, "r": 15, "t": 70, "b": 15},
    )
    return fig


def make_bar_chart(table, column, title, y_label):
    fig = px.bar(
        table,
        x="Controller",
        y=column,
        color="Controller",
        color_discrete_map=COLORS,
        title=title,
        text_auto=".4g",
    )
    fig.update_layout(
        xaxis_title="Controller",
        yaxis_title=y_label,
        showlegend=False,
        margin={"l": 15, "r": 15, "t": 60, "b": 15},
    )
    return fig


def to_csv_bytes(dataframe):
    return dataframe.to_csv(index=False).encode("utf-8")


# ---------------------------------------------------------------------
# Header and service status
# ---------------------------------------------------------------------

st.title("Active Boring-Bar Vibration Control")
st.caption(
    "Simulation dashboard | No Control vs PID vs Soft Actor-Critic (SAC)"
)

with st.sidebar:
    st.header("Simulation settings")

    amplitude_n = st.slider(
        "Disturbance amplitude (N)",
        min_value=0.0,
        max_value=3.0,
        value=1.0,
        step=0.1,
    )

    frequency_hz = st.slider(
        "Disturbance frequency (Hz)",
        min_value=0.0,
        max_value=250.0,
        value=174.0,
        step=1.0,
    )

    phase_rad = st.slider(
        "Disturbance phase (rad)",
        min_value=-3.14,
        max_value=3.14,
        value=0.0,
        step=0.1,
    )

    duration_s = st.slider(
        "Simulation duration (s)",
        min_value=0.02,
        max_value=0.5,
        value=0.2,
        step=0.02,
    )

    st.divider()
    st.caption(
        "These are simulated disturbance parameters, not live machine inputs."
    )

try:
    health = api_get("/health")
    st.sidebar.success("API connected")
    model_status = health.get("sac_model", {})
    if model_status.get("model_exists"):
        st.sidebar.success("SAC model file found")
    else:
        st.sidebar.warning("SAC model file not found")
        st.sidebar.caption(model_status.get("message", ""))

except requests.RequestException as exc:
    st.error(
        "Cannot connect to the API. Start the FastAPI server in another "
        "terminal, then rerun this dashboard."
    )
    st.code(
        r".venv\Scripts\python.exe -m uvicorn api:app --reload",
        language="bat",
    )
    st.caption(f"Connection detail: {exc}")
    st.stop()


# ---------------------------------------------------------------------
# Run simulations
# ---------------------------------------------------------------------

st.subheader("Run comparison")

st.write(
    "All three controllers are evaluated with the same disturbance "
    "amplitude, frequency, phase, and duration."
)

run_clicked = st.button(
    "Run all three controllers",
    type="primary",
    use_container_width=True,
)

if run_clicked:
    payload = {
        "amplitude_n": amplitude_n,
        "frequency_hz": frequency_hz,
        "phase_rad": phase_rad,
        "duration_s": duration_s,
    }

    try:
        with st.spinner("Running simulations..."):
            traces = {}
            metrics_results = []

            for controller in CONTROLLERS:
                result = api_post(
                    "/simulate",
                    {"controller": controller, **payload},
                )
                traces[controller] = pd.DataFrame(result["timeseries"])
                metrics_results.append({
                    "controller": controller,
                    **result["metrics"],
                })

            st.session_state["traces"] = traces
            st.session_state["metrics_results"] = metrics_results
            st.session_state["simulation_inputs"] = payload

    except requests.HTTPError as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        st.error(f"API returned an error: {detail}")
    except requests.RequestException as exc:
        st.error(f"Simulation request failed: {exc}")
    except (KeyError, ValueError, TypeError) as exc:
        st.error(f"Unexpected API response: {exc}")


if "traces" not in st.session_state:
    st.info("Set the parameters in the sidebar and click Run all three controllers.")
    st.stop()


traces = st.session_state["traces"]
metrics_results = st.session_state["metrics_results"]
run_inputs = st.session_state["simulation_inputs"]
table = make_comparison_table(metrics_results)

# ---------------------------------------------------------------------
# Context and key performance indicators
# ---------------------------------------------------------------------

st.divider()
st.subheader("Simulation condition")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Disturbance amplitude", f"{run_inputs['amplitude_n']:.2f} N")
c2.metric("Frequency", f"{run_inputs['frequency_hz']:.1f} Hz")
c3.metric("Phase", f"{run_inputs['phase_rad']:.2f} rad")
c4.metric("Duration", f"{run_inputs['duration_s']:.2f} s")

frequency = run_inputs["frequency_hz"]
if 150 <= frequency <= 200:
    st.success("Frequency is within the SAC training range (150–200 Hz).")
else:
    st.warning(
        "Frequency is outside the SAC training range (150–200 Hz). "
        "Treat SAC performance here as an out-of-range test."
    )

baseline_rows = table[table["Controller"] == "No control"]
if not baseline_rows.empty:
    baseline_rms = baseline_rows.iloc[0]["Steady RMS (µm)"]
    controller_rows = table[table["Controller"] != "No control"]

    if np.isfinite(baseline_rms) and baseline_rms > 0:
        cols = st.columns(max(1, len(controller_rows)))
        for col, (_, row) in zip(cols, controller_rows.iterrows()):
            rms = row["Steady RMS (µm)"]
            reduction = 100 * (1 - rms / baseline_rms) if np.isfinite(rms) else np.nan
            col.metric(
                f"{row['Controller']} steady RMS reduction",
                f"{reduction:.2f}%" if np.isfinite(reduction) else "N/A",
                help="Positive means lower steady-state RMS than No Control; negative means worse.",
            )

# ---------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------

tab_vibration, tab_force, tab_metrics, tab_data = st.tabs([
    "Vibration response",
    "Force response",
    "Performance metrics",
    "Raw data & downloads",
])

with tab_vibration:
    st.plotly_chart(
        make_trace_plot(
            traces,
            "displacement_um",
            "Tool-mode displacement response",
            "Displacement (µm)",
        ),
        use_container_width=True,
    )

    st.plotly_chart(
        make_trace_plot(
            traces,
            "velocity_mm_s",
            "Vibration velocity response",
            "Velocity (mm/s)",
        ),
        use_container_width=True,
    )

    st.caption(
        "The displacement is the simulated modal coordinate expressed in "
        "micrometres. It is not necessarily physical tool-tip displacement "
        "without a calibrated mode-shape scale."
    )

with tab_force:
    st.plotly_chart(
        make_force_disturbance_plot(traces),
        use_container_width=True,
    )

    st.caption(
        "Control force is the simulated generalized force. It does not model "
        "real actuator bandwidth, delay, or hardware dynamics."
    )

with tab_metrics:
    st.subheader("Metric table")
    st.dataframe(
        table.style.format({
            "Steady RMS (µm)": "{:.4f}",
            "Overall RMS (µm)": "{:.4f}",
            "Peak vibration (µm)": "{:.4f}",
            "RMS force (N)": "{:.4f}",
            "Peak force (N)": "{:.4f}",
            "Saturation (%)": "{:.2f}",
        }, na_rep="N/A"),
        use_container_width=True,
        hide_index=True,
    )

    left, right = st.columns(2)

    with left:
        st.plotly_chart(
            make_bar_chart(
                table,
                "Steady RMS (µm)",
                "Steady-state RMS vibration",
                "RMS (µm)",
            ),
            use_container_width=True,
        )

        st.plotly_chart(
            make_bar_chart(
                table,
                "Peak vibration (µm)",
                "Peak absolute displacement",
                "Peak (µm)",
            ),
            use_container_width=True,
        )

    with right:
        st.plotly_chart(
            make_bar_chart(
                table,
                "RMS force (N)",
                "RMS control effort",
                "RMS force (N)",
            ),
            use_container_width=True,
        )

        st.plotly_chart(
            make_bar_chart(
                table,
                "Saturation (%)",
                "Actuator saturation",
                "Saturated samples (%)",
            ),
            use_container_width=True,
        )

    st.info(
        "Lower vibration is desirable, but compare it alongside control effort "
        "and saturation. These metrics characterize this simulated scenario "
        "only; one run is not a robustness study."
    )

with tab_data:
    selected_controller = st.selectbox(
        "Select controller trace",
        CONTROLLERS,
    )

    selected_data = traces[selected_controller]
    st.dataframe(
        selected_data,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download selected time series (CSV)",
        data=to_csv_bytes(selected_data),
        file_name=f"{selected_controller.lower().replace(' ', '_')}_trace.csv",
        mime="text/csv",
    )

    st.download_button(
        "Download all controller metrics (CSV)",
        data=to_csv_bytes(table),
        file_name="controller_comparison_metrics.csv",
        mime="text/csv",
    )

    combined = []
    for controller, data in traces.items():
        copy = data.copy()
        copy.insert(0, "controller", controller)
        combined.append(copy)

    all_traces = pd.concat(combined, ignore_index=True)
    st.download_button(
        "Download all time series (CSV)",
        data=to_csv_bytes(all_traces),
        file_name="all_controller_traces.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------------
# Model scope and limitations
# ---------------------------------------------------------------------

with st.expander("Model assumptions and limitations"):
    st.markdown(
        """
        - This is a **single-mode oscillator with a harmonic disturbance**.
        - The SAC policy is loaded from a previously trained model; this
          dashboard does not train it.
        - PID uses fixed gains; SAC uses a learned policy.
        - The current evaluation does not add sensor noise or randomized
          plant parameters within each run.
        - Results do not establish performance on regenerative chatter,
          a physical boring bar, or real actuator hardware.
        - A single manually selected condition is not sufficient to claim
          robustness or general superiority.
        """
    )

st.caption(
    "Educational simulation only — not connected to physical machinery."
)
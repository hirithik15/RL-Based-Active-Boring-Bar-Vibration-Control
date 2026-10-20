
"""Streamlit dashboard for machining-aware boring-bar simulation."""

import requests
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Boring-Bar Vibration Control",
    page_icon="📉",
    layout="wide",
)

CONTROLLERS = ["No control", "PID", "SAC"]
DEFAULT_API_URL = "http://127.0.0.1:8000"


# ------------------------------------------------------------------
# API helpers
# ------------------------------------------------------------------

def api_url():
    return st.session_state.get("api_base", DEFAULT_API_URL).strip().rstrip("/")


def request_api(method, path, payload=None, timeout=90):
    url = f"{api_url()}{path}"

    try:
        if method == "GET":
            response = requests.get(url, timeout=5)
        else:
            response = requests.post(
                url,
                json=payload,
                timeout=(5, timeout),
            )

        if not response.ok:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text

            raise RuntimeError(f"API HTTP {response.status_code}: {detail}")

        try:
            result = response.json()
        except ValueError as exc:
            raise RuntimeError("API returned invalid JSON.") from exc

        if not isinstance(result, dict):
            raise RuntimeError("API response was not a JSON object.")

        return result

    except requests.Timeout as exc:
        raise RuntimeError("API request timed out. Check the server terminal.") from exc
    except requests.ConnectionError as exc:
        raise RuntimeError(
            f"Cannot connect to {url}. Start FastAPI in a separate terminal."
        ) from exc


def validate_result(result, controller):
    if "metrics" not in result or not isinstance(result["metrics"], dict):
        raise RuntimeError(f"{controller}: API response has no metrics object.")

    records = result.get("timeseries")
    if not isinstance(records, list) or len(records) < 2:
        raise RuntimeError(f"{controller}: API returned no usable time series.")

    frame = pd.DataFrame(records)
    if frame.empty:
        raise RuntimeError(f"{controller}: empty time-series data.")

    numeric_columns = frame.select_dtypes(include="number").columns.tolist()
    if not numeric_columns:
        raise RuntimeError(f"{controller}: no numeric time-series columns.")

    if not np.isfinite(frame[numeric_columns].to_numpy(dtype=float)).all():
        raise RuntimeError(f"{controller}: time series contains NaN or infinity.")

    return frame


def csv_bytes(frame):
    return frame.to_csv(index=False).encode("utf-8")


def metric_number(metrics, key):
    try:
        value = float(metrics.get(key))
        return value if np.isfinite(value) else np.nan
    except (TypeError, ValueError):
        return np.nan


def comparison_frame(results):
    rows = []
    for item in results:
        rows.append({"Controller": item["controller"], **{
            key: value
            for key, value in item.items()
            if key != "controller"
        }})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Header and sidebar
# ------------------------------------------------------------------

st.title("Active Boring-Bar Vibration Control")
st.caption(
    "Machining-aware force simulation · No Control vs PID vs SAC"
)

with st.sidebar:
    st.header("API connection")
    st.text_input(
        "FastAPI URL",
        value=st.session_state.get("api_base", DEFAULT_API_URL),
        key="api_base",
    )

    st.divider()
    st.header("Machining parameters")

    spindle_rpm = st.number_input(
        "Spindle speed (RPM)",
        min_value=1.0,
        max_value=100000.0,
        value=9000.0,
        step=100.0,
    )

    feed_mm_rev = st.number_input(
        "Feed (mm/rev)",
        min_value=0.001,
        max_value=10.0,
        value=0.1,
        step=0.01,
        format="%.3f",
    )

    depth_of_cut_mm = st.number_input(
        "Depth of cut (mm)",
        min_value=0.001,
        max_value=20.0,
        value=0.5,
        step=0.1,
    )

    specific_cutting_force_n_mm2 = st.number_input(
        "Specific cutting force (N/mm²)",
        min_value=1.0,
        max_value=10000.0,
        value=1800.0,
        step=100.0,
    )

    st.divider()
    st.header("Force-model settings")

    force_fluctuation_fraction = st.slider(
        "Fluctuation fraction",
        min_value=0.0,
        max_value=2.0,
        value=0.2,
        step=0.05,
    )

    random_force_fraction = st.slider(
        "Random force fraction",
        min_value=0.0,
        max_value=2.0,
        value=0.08,
        step=0.01,
    )

    modal_participation = st.slider(
        "Modal participation",
        min_value=0.001,
        max_value=1.0,
        value=0.02,
        step=0.005,
        format="%.3f",
    )

    transient_rate_hz = st.slider(
        "Transient rate (events/s)",
        min_value=0.0,
        max_value=100.0,
        value=3.0,
        step=0.5,
    )

    transient_fraction = st.slider(
        "Transient strength",
        min_value=0.0,
        max_value=5.0,
        value=0.5,
        step=0.1,
    )

    phase_rad = st.slider(
        "Phase (rad)",
        min_value=-3.14,
        max_value=3.14,
        value=0.0,
        step=0.1,
    )

    duration_s = st.slider(
        "Duration (s)",
        min_value=0.02,
        max_value=2.0,
        value=0.2,
        step=0.02,
    )

    seed = st.number_input(
        "Random seed",
        min_value=0,
        max_value=2147483647,
        value=12345,
        step=1,
    )


payload = {
    "spindle_rpm": float(spindle_rpm),
    "feed_mm_rev": float(feed_mm_rev),
    "depth_of_cut_mm": float(depth_of_cut_mm),
    "specific_cutting_force_n_mm2": float(specific_cutting_force_n_mm2),
    "force_fluctuation_fraction": float(force_fluctuation_fraction),
    "random_force_fraction": float(random_force_fraction),
    "modal_participation": float(modal_participation),
    "transient_rate_hz": float(transient_rate_hz),
    "transient_fraction": float(transient_fraction),
    "phase_rad": float(phase_rad),
    "duration_s": float(duration_s),
    "seed": int(seed),
}


# ------------------------------------------------------------------
# Health status
# ------------------------------------------------------------------

api_healthy = False

try:
    health = request_api("GET", "/health")
    api_healthy = health.get("status") == "ok"
    st.sidebar.success("API connected")

    model_info = health.get("sac_model", {})
    if model_info.get("model_exists"):
        st.sidebar.success("SAC model file found")
    else:
        st.sidebar.warning("SAC model file not found")
        st.sidebar.caption(model_info.get("message", "Check the model path."))

except RuntimeError as exc:
    st.sidebar.error("API unavailable")
    st.error(str(exc))
    st.code(
        "python -m uvicorn api:app --reload --host 127.0.0.1 --port 8000",
        language="bash",
    )


# ------------------------------------------------------------------
# Simulation controls
# ------------------------------------------------------------------

st.subheader("Machining condition")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Spindle speed", f"{spindle_rpm:,.0f} RPM")
c2.metric("Feed", f"{feed_mm_rev:.3f} mm/rev")
c3.metric("Depth of cut", f"{depth_of_cut_mm:.3f} mm")
c4.metric("Duration", f"{duration_s:.2f} s")

st.write(
    "Run the same machining condition for each controller. "
    "The random seed is held constant to support a fair comparison, "
    "provided the force model uses the seed consistently."
)

run_one_col, run_all_col = st.columns(2)

with run_one_col:
    selected_controller = st.selectbox(
        "Single controller",
        CONTROLLERS,
    )

    run_single = st.button(
        "Run selected controller",
        use_container_width=True,
        disabled=not api_healthy,
    )

with run_all_col:
    run_all = st.button(
        "Compare all controllers",
        type="primary",
        use_container_width=True,
        disabled=not api_healthy,
    )


def run_simulation(controller):
    result = request_api(
        "POST",
        "/simulate",
        {"controller": controller, **payload},
        timeout=120,
    )
    frame = validate_result(result, controller)
    return frame, result["metrics"], result


if run_single or run_all:
    controllers_to_run = CONTROLLERS if run_all else [selected_controller]
    new_traces = {}
    new_metrics = []
    new_responses = {}

    try:
        progress = st.progress(0, text="Starting simulation...")

        for index, controller in enumerate(controllers_to_run):
            progress.progress(
                index / len(controllers_to_run),
                text=f"Running {controller}...",
            )

            frame, metrics, raw_result = run_simulation(controller)
            new_traces[controller] = frame
            new_metrics.append({"controller": controller, **metrics})
            new_responses[controller] = raw_result

        # Preserve prior results if any controller in this batch fails.
        st.session_state["traces"] = {
            **st.session_state.get("traces", {}),
            **new_traces,
        }
        st.session_state["metrics"] = {
            **st.session_state.get("metrics", {}),
            **{row["controller"]: row for row in new_metrics},
        }
        st.session_state["raw_responses"] = {
            **st.session_state.get("raw_responses", {}),
            **new_responses,
        }
        st.session_state["last_inputs"] = payload.copy()

        progress.progress(1.0, text="Simulation completed.")
        st.success("Simulation completed successfully.")

    except RuntimeError as exc:
        st.error(str(exc))
        st.caption("Check the FastAPI terminal for the underlying traceback.")

    finally:
        if "progress" in locals():
            progress.empty()


if "traces" not in st.session_state:
    st.info("Run a controller simulation to view its results.")
    st.stop()


traces = st.session_state["traces"]
metrics_by_controller = st.session_state.get("metrics", {})
available_controllers = [
    name for name in CONTROLLERS if name in traces
]


# ------------------------------------------------------------------
# Results and plots
# ------------------------------------------------------------------

st.divider()
st.subheader("Simulation results")

metric_rows = []
for controller in available_controllers:
    metrics = metrics_by_controller.get(controller, {})
    metric_rows.append({"Controller": controller, **metrics})

metrics_table = pd.DataFrame(metric_rows)

if not metrics_table.empty:
    st.dataframe(metrics_table, use_container_width=True, hide_index=True)

tab_displacement, tab_velocity, tab_force, tab_data = st.tabs([
    "Displacement",
    "Velocity",
    "Force",
    "Data & downloads",
])


def plot_available_column(column, title, y_label):
    fig = go.Figure()

    for controller in available_controllers:
        frame = traces[controller]
        if "time_s" not in frame.columns or column not in frame.columns:
            continue

        fig.add_trace(
            go.Scatter(
                x=frame["time_s"],
                y=frame[column],
                mode="lines",
                name=controller,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title=y_label,
        hovermode="x unified",
    )
    return fig


with tab_displacement:
    if any("displacement_um" in traces[c].columns for c in available_controllers):
        st.plotly_chart(
            plot_available_column(
                "displacement_um",
                "Tool-mode displacement",
                "Displacement (µm)",
            ),
            use_container_width=True,
        )
    else:
        st.warning("The API response does not contain displacement_um.")

with tab_velocity:
    if any("velocity_mm_s" in traces[c].columns for c in available_controllers):
        st.plotly_chart(
            plot_available_column(
                "velocity_mm_s",
                "Tool-mode velocity",
                "Velocity (mm/s)",
            ),
            use_container_width=True,
        )
    else:
        st.warning("The API response does not contain velocity_mm_s.")

with tab_force:
    force_columns = [
        ("disturbance_force_n", "Machining / disturbance force"),
        ("cutting_force_n", "Cutting force"),
        ("control_force_n", "Controller force"),
    ]

    plotted = False
    for column, title in force_columns:
        if any(column in traces[c].columns for c in available_controllers):
            st.plotly_chart(
                plot_available_column(column, title, "Force (N)"),
                use_container_width=True,
            )
            plotted = True

    if not plotted:
        st.warning("No recognized force columns were returned by the API.")

# Add "Performance metrics" to your tabs
tab_displacement, tab_velocity, tab_force, tab_metrics, tab_data = st.tabs([
    "Displacement",
    "Velocity",
    "Force",
    "Performance metrics",
    "Data & downloads",
])

# Add this block after the existing force plots
with tab_metrics:
    st.subheader("Vibration and controller performance")

    metric_rows = []
    for controller in available_controllers:
        metrics = metrics_by_controller.get(controller, {})
        metric_rows.append({
            "Controller": controller,
            "Steady RMS (µm)": metric_number(
                metrics, "steady_rms_vibration_um"
            ),
            "Overall RMS (µm)": metric_number(
                metrics, "rms_vibration_um"
            ),
            "Peak vibration (µm)": metric_number(
                metrics, "peak_vibration_um"
            ),
            "RMS control force (N)": metric_number(
                metrics, "rms_control_force_n"
            ),
            "Saturation (%)": 100 * metric_number(
                metrics, "actuator_saturation_fraction"
            ),
        })

    metrics_table = pd.DataFrame(metric_rows)

    st.dataframe(
        metrics_table,
        use_container_width=True,
        hide_index=True,
    )

    def rms_bar_chart(column, title, y_label):
        valid = metrics_table.dropna(subset=[column])

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=valid["Controller"],
            y=valid[column],
            text=valid[column].map(
                lambda value: f"{value:.4g}"
            ),
            textposition="outside",
        ))
        fig.update_layout(
            title=title,
            xaxis_title="Controller",
            yaxis_title=y_label,
            showlegend=False,
        )
        return fig

    left, right = st.columns(2)

    with left:
        st.plotly_chart(
            rms_bar_chart(
                "Steady RMS (µm)",
                "Steady-state RMS displacement",
                "RMS displacement (µm)",
            ),
            use_container_width=True,
        )

        st.plotly_chart(
            rms_bar_chart(
                "Overall RMS (µm)",
                "Overall RMS displacement",
                "RMS displacement (µm)",
            ),
            use_container_width=True,
        )

    with right:
        st.plotly_chart(
            rms_bar_chart(
                "Peak vibration (µm)",
                "Peak absolute displacement",
                "Displacement (µm)",
            ),
            use_container_width=True,
        )

        st.plotly_chart(
            rms_bar_chart(
                "RMS control force (N)",
                "RMS controller effort",
                "Force (N)",
            ),
            use_container_width=True,
        )

    st.caption(
        "Lower RMS displacement indicates less vibration. Compare it with "
        "control effort and actuator saturation before judging a controller."
    )
    
with tab_data:
    controller_to_view = st.selectbox(
        "Trace to inspect",
        available_controllers,
    )

    frame = traces[controller_to_view]
    st.dataframe(frame, use_container_width=True, hide_index=True)

    st.download_button(
        "Download selected time series",
        data=csv_bytes(frame),
        file_name=f"{controller_to_view.lower().replace(' ', '_')}_timeseries.csv",
        mime="text/csv",
    )

    if not metrics_table.empty:
        st.download_button(
            "Download metrics",
            data=csv_bytes(metrics_table),
            file_name="controller_metrics.csv",
            mime="text/csv",
        )

    all_frames = []
    for controller in available_controllers:
        copy = traces[controller].copy()
        copy.insert(0, "controller", controller)
        all_frames.append(copy)

    st.download_button(
        "Download all available time series",
        data=csv_bytes(pd.concat(all_frames, ignore_index=True)),
        file_name="all_controller_timeseries.csv",
        mime="text/csv",
    )


with st.expander("Model limitations"):
    st.markdown(
        """
        - Results are from a simulation, not a physical machine.
        - The force model is only as reliable as its assumptions and
          parameter calibration.
        - Use the same seed and machining conditions for comparisons.
        - A single simulation does not establish robustness.
        - Confirm that the SAC model was trained with an observation space,
          action space, plant, and force model compatible with this service.
        """
    )

st.caption("Simulation only — no connection to physical machinery.")

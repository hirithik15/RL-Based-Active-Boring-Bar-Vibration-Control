
"""Run fair controller comparisons and save quantitative results."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import SAC

from config import PLANT, TRAINING, EVALUATION, DisturbanceConfig
from models import SingleModePlant, SinusoidalCuttingForce
from controllers import (
    NoControl,
    PIDController,
    ActiveDampingController,
)


def simulate_controller(
    controller_name,
    disturbance_config,
    duration_s,
    model=None,
):
    """Simulate one controller using the shared plant integration method."""
    if duration_s <= 0:
        raise ValueError("duration_s must be positive.")

    plant = SingleModePlant(PLANT)
    plant.reset(0.0, 0.0)
    force_model = SinusoidalCuttingForce(disturbance_config)

    if controller_name == "No control":
        controller = NoControl()

    elif controller_name == "Active damping":
        controller = ActiveDampingController(
            velocity_gain=20.0,
            plant_config=PLANT,
        )

    elif controller_name == "PID":
        controller = PIDController(plant_config=PLANT)

    elif controller_name == "SAC":
        if model is None:
            raise ValueError("A trained SAC model is required.")
        controller = None

    else:
        raise ValueError(f"Unknown controller: {controller_name}")

    if controller is not None:
        controller.reset()

    dt = PLANT.dt_s
    n_steps = int(round(duration_s / dt))
    previous_action = 0.0
    records = []

    for step_index in range(n_steps):
        time_s = step_index * dt
        state = plant.state
        disturbance_n = force_model(time_s)

        if controller_name == "SAC":
            observation = np.asarray(
                [
                    state.displacement_m / TRAINING.displacement_scale_m,
                    state.velocity_m_s / TRAINING.velocity_scale_m_s,
                    previous_action,
                ],
                dtype=np.float32,
            )
            action, _ = model.predict(
                observation,
                deterministic=True,
            )
            normalized_action = float(
                np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0)
            )
            control_n = normalized_action * PLANT.max_control_force_n
            previous_action = normalized_action
        else:
            control_n = float(
                controller.act(
                    state.displacement_m,
                    state.velocity_m_s,
                    dt,
                )
            )
            control_n = float(
                np.clip(
                    control_n,
                    -PLANT.max_control_force_n,
                    PLANT.max_control_force_n,
                )
            )

        next_state = plant.step(
            disturbance_n=disturbance_n,
            control_n=control_n,
        )

        records.append(
            {
                "time_s": time_s + dt,
                "displacement_um": next_state.displacement_m * 1e6,
                "velocity_mm_s": next_state.velocity_m_s * 1e3,
                "disturbance_force_n": disturbance_n,
                "control_force_n": control_n,
            }
        )

    return pd.DataFrame(records)


def calculate_metrics(data):
    """Calculate full-record and steady-window vibration/control metrics."""
    displacement = data["displacement_um"].to_numpy()
    force = data["control_force_n"].to_numpy()

    if len(data) == 0:
        raise ValueError("Cannot calculate metrics for empty data.")

    # Discard the first 20% as a transient window for the steady metric.
    start = min(int(0.2 * len(data)), len(data) - 1)
    steady_displacement = displacement[start:]

    force_limit = PLANT.max_control_force_n
    saturation = np.abs(force) >= 0.99 * force_limit

    return {
        "rms_vibration_um": float(np.sqrt(np.mean(displacement**2))),
        "peak_vibration_um": float(np.max(np.abs(displacement))),
        "steady_rms_vibration_um": float(
            np.sqrt(np.mean(steady_displacement**2))
        ),
        "rms_control_force_n": float(np.sqrt(np.mean(force**2))),
        "peak_control_force_n": float(np.max(np.abs(force))),
        "actuator_saturation_fraction": float(np.mean(saturation)),
    }


def _save_plots(condition_name, condition_data, output_dir):
    """Save overlay, SAC-only zoom, and controller-force plots."""
    output_dir = Path(output_dir)
    time_ms = condition_data["No control"]["time_s"].to_numpy() * 1000.0

    # Overlay can be dominated by the uncontrolled response.
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, data in condition_data.items():
        ax.plot(
            time_ms,
            data["displacement_um"],
            label=name,
            linewidth=1.1,
        )
    ax.set_title(f"Displacement comparison — {condition_name}")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Modal-coordinate displacement (µm)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        output_dir / f"displacement_{condition_name}_overlay.png",
        dpi=160,
    )
    plt.close(fig)

    # A separate SAC plot avoids hiding its trace on the overlay scale.
    sac_data = condition_data["SAC"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        sac_data["time_s"] * 1000.0,
        sac_data["displacement_um"],
        linewidth=1.2,
        label="SAC",
    )
    ax.set_title(f"SAC displacement detail — {condition_name}")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Modal-coordinate displacement (µm)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        output_dir / f"displacement_{condition_name}_sac_zoom.png",
        dpi=160,
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, data in condition_data.items():
        ax.plot(
            data["time_s"] * 1000.0,
            data["control_force_n"],
            label=name,
            linewidth=1.0,
        )
    ax.axhline(
        PLANT.max_control_force_n,
        linestyle="--",
        linewidth=0.8,
        label="Force limit",
    )
    ax.axhline(
        -PLANT.max_control_force_n,
        linestyle="--",
        linewidth=0.8,
    )
    ax.set_title(f"Control force — {condition_name}")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Control force (N)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"control_{condition_name}.png", dpi=160)
    plt.close(fig)


def run_experiments(
    model_path="results/sac_vibration_model.zip",
    output_dir="results",
):
    """Evaluate No Control, PID, and SAC on nominal and shifted inputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = SAC.load(str(model_path))

    conditions = {
        "nominal": DisturbanceConfig(
            amplitude_n=EVALUATION.nominal_amplitude_n,
            frequency_hz=EVALUATION.nominal_frequency_hz,
            phase_rad=0.0,
        ),
        "unseen": DisturbanceConfig(
            amplitude_n=EVALUATION.unseen_amplitude_n,
            frequency_hz=EVALUATION.unseen_frequency_hz,
            phase_rad=EVALUATION.test_phase_rad,
        ),
    }

    all_metrics = []
    all_timeseries = []

    for condition_name, disturbance in conditions.items():
        condition_data = {}

        for controller_name in ("No control",
                                "Active damping",
                                "PID",
                                "SAC",
                                ):
            data = simulate_controller(
                controller_name=controller_name,
                disturbance_config=disturbance,
                duration_s=EVALUATION.duration_s,
                model=model if controller_name == "SAC" else None,
            )
            data.insert(0, "controller", controller_name)
            data.insert(0, "condition", condition_name)
            condition_data[controller_name] = data

            metrics = calculate_metrics(data)
            metrics["condition"] = condition_name
            metrics["controller"] = controller_name
            all_metrics.append(metrics)
            all_timeseries.append(data)

        _save_plots(condition_name, condition_data, output_dir)

    metrics_frame = pd.DataFrame(all_metrics)
    timeseries_frame = pd.concat(all_timeseries, ignore_index=True)

    metrics_path = output_dir / "controller_metrics.csv"
    timeseries_path = output_dir / "controller_timeseries.csv"

    metrics_frame.to_csv(metrics_path, index=False)
    timeseries_frame.to_csv(timeseries_path, index=False)

    print("\nController metrics:")
    print(metrics_frame.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print(f"\nMetrics saved to: {metrics_path}")
    print(f"Time series saved to: {timeseries_path}")
    print(f"Plots saved under: {output_dir.resolve()}")

    return metrics_frame


if __name__ == "__main__":
    run_experiments()

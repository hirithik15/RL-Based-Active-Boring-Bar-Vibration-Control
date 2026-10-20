
"""Train-independent evaluation of controllers under machining-aware forces."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import SAC

from config import PLANT, TRAINING, EVALUATION, DisturbanceConfig, MachiningConfig
from models import SingleModePlant, SinusoidalCuttingForce
from cutting_force_model import MachiningCuttingForce
from controllers import (
    NoControl,
    PIDController,
    ActiveDampingController,
)


def simulate_controller(
    controller_name,
    disturbance_config=None,
    duration_s=0.2,
    model=None,
    machining_config=None,
    seed=12345,
):
    """Simulate one controller with a shared plant and selected force model.

    Pass machining_config to use the new stochastic/regenerative model.
    The legacy disturbance_config argument remains for older scripts.
    """
    if duration_s <= 0:
        raise ValueError("duration_s must be positive.")

    plant = SingleModePlant(PLANT)
    plant.reset(0.0, 0.0)

    if machining_config is not None:
        force_model = MachiningCuttingForce(
            config=machining_config,
            dt_s=PLANT.dt_s,
            rng=np.random.default_rng(seed),
        )
    else:
        if disturbance_config is None:
            disturbance_config = DisturbanceConfig()
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

        if machining_config is not None:
            disturbance_n = force_model(
                time_s=time_s,
                displacement_m=state.displacement_m,
            )
        else:
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
            action, _ = model.predict(observation, deterministic=True)
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
    """Calculate vibration and control-effort metrics."""
    if len(data) == 0:
        raise ValueError("Cannot calculate metrics for empty data.")

    displacement = data["displacement_um"].to_numpy()
    force = data["control_force_n"].to_numpy()

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
    """Save displacement, SAC detail, control force, and disturbance plots."""
    output_dir = Path(output_dir)
    time_ms = (
        condition_data["No control"]["time_s"].to_numpy() * 1000.0
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, data in condition_data.items():
        ax.plot(
            data["time_s"] * 1000.0,
            data["displacement_um"],
            label=name,
            linewidth=1.0,
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

    sac_data = condition_data["SAC"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        sac_data["time_s"] * 1000.0,
        sac_data["displacement_um"],
        linewidth=1.1,
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

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, data in condition_data.items():
        ax.plot(
            data["time_s"] * 1000.0,
            data["disturbance_force_n"],
            label=name,
            linewidth=0.9,
        )
    ax.set_title(f"Cutting force — {condition_name}")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Generalized cutting force (N)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        output_dir / f"cutting_force_{condition_name}.png",
        dpi=160,
    )
    plt.close(fig)


def run_experiments(
    model_path="results/sac_machining_aware_v1.zip",
    output_dir="results",
):
    """Compare all controllers on fixed nominal and shifted machining cases."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = SAC.load(str(model_path))

    common = dict(
        feed_mm_rev=EVALUATION.feed_mm_rev,
        depth_of_cut_mm=EVALUATION.depth_of_cut_mm,
        specific_cutting_force_n_mm2=(
            EVALUATION.specific_cutting_force_n_mm2
        ),
        force_fluctuation_fraction=(
            EVALUATION.force_fluctuation_fraction
        ),
        random_force_fraction=EVALUATION.random_force_fraction,
        modal_participation=EVALUATION.modal_participation,
        transient_rate_hz=EVALUATION.transient_rate_hz,
        transient_fraction=EVALUATION.transient_fraction,
        transient_decay_s=0.003,
        regenerative_feedback_sign=1.0,
    )

    conditions = {
        "nominal": MachiningConfig(
            spindle_rpm=EVALUATION.nominal_spindle_rpm,
            phase_rad=0.0,
            **common,
        ),
        "unseen": MachiningConfig(
            spindle_rpm=EVALUATION.unseen_spindle_rpm,
            phase_rad=EVALUATION.test_phase_rad,
            # Increased disturbance uncertainty for the shifted test.
            force_fluctuation_fraction=min(
                0.60, EVALUATION.force_fluctuation_fraction * 1.35
            ),
            random_force_fraction=min(
                0.25, EVALUATION.random_force_fraction * 1.5
            ),
            transient_rate_hz=EVALUATION.transient_rate_hz * 1.5,
            **{
                k: v for k, v in common.items()
                if k not in {
                    "force_fluctuation_fraction",
                    "random_force_fraction",
                    "transient_rate_hz",
                }
            },
        ),
    }

    all_metrics = []
    all_timeseries = []

    for condition_index, (condition_name, machining) in enumerate(
        conditions.items()
    ):
        print(f"\nCondition: {condition_name}")
        print("Machining and force-model parameters:")
        for key, value in machining.__dict__.items():
            print(f"  {key}: {value}")

        condition_data = {}

        for controller_name in (
            "No control",
            "Active damping",
            "PID",
            "SAC",
        ):
            data = simulate_controller(
                controller_name=controller_name,
                machining_config=machining,
                duration_s=EVALUATION.duration_s,
                model=model if controller_name == "SAC" else None,
                # Same seed across controllers: same random sequence for
                # noise and event generation, though feedback differs.
                seed=1000 + condition_index,
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
    print(
        metrics_frame.to_string(
            index=False,
            float_format=lambda x: f"{x:.6g}",
        )
    )
    print(f"\nMetrics saved to: {metrics_path}")
    print(f"Time series saved to: {timeseries_path}")
    print(f"Plots saved under: {output_dir.resolve()}")

    return metrics_frame


if __name__ == "__main__":
    run_experiments()

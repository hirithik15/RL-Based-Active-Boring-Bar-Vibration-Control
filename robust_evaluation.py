
"""Compare No control, tuned PID, and SAC across 84 test conditions."""

from pathlib import Path
import math
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import SAC

from config import PLANT, TRAINING, DisturbanceConfig
from controllers import NoControl, PIDController
from models import SingleModePlant, SinusoidalCuttingForce


# Keep PID fixed throughout testing; do not tune on the test conditions.
PID_GAINS_PATH = Path("results/tuned_pid.json")
MODEL_PATH = Path("results/sac_vibration_model_v2.zip")
OUT = Path("results")
DURATION_S = 0.2
STEADY_FRACTION = 0.20

AMPLITUDES = [0.5, 1.0, 1.5]
FREQUENCIES = [150, 160, 174, 185, 200, 205, 215]
PHASES = [0, math.pi / 2, math.pi, 3 * math.pi / 2]


def make_conditions():
    """84 combinations: 3 amplitudes x 7 frequencies x 4 phases."""
    result = []
    for amplitude in AMPLITUDES:
        for frequency in FREQUENCIES:
            for phase in PHASES:
                result.append({
                    "amplitude_n": amplitude,
                    "frequency_hz": frequency,
                    "phase_rad": phase,
                    "frequency_group": (
                        "within_training_range"
                        if TRAINING.force_frequency_min_hz
                        <= frequency <= TRAINING.force_frequency_max_hz
                        else "outside_training_range"
                    ),
                })
    return result

def simulate(name, condition, sac=None, pid_gains=None):
    plant = SingleModePlant(PLANT)
    plant.reset(0.0, 0.0)

    disturbance = SinusoidalCuttingForce(DisturbanceConfig(
        amplitude_n=condition["amplitude_n"],
        frequency_hz=condition["frequency_hz"],
        phase_rad=condition["phase_rad"],
    ))

    pid = None
    if name == "PID":
        if pid_gains is None:
            raise ValueError(
                "PID gains missing. Run tune_pid.py first."
            )
        pid = PIDController(
            kp=pid_gains["kp"],
            ki=pid_gains["ki"],
            kd=pid_gains["kd"],
            plant_config=PLANT,
        )
        pid.reset()

    n = round(DURATION_S / PLANT.dt_s)
    steady_start = int(n * STEADY_FRACTION)
    x = np.zeros(n)
    u = np.zeros(n)
    previous_action = 0.0

    for i in range(n):
        state = plant.state

        if name == "No control":
            force = 0.0
        elif name == "PID":
            force = pid.act(
                state.displacement_m,
                state.velocity_m_s,
                PLANT.dt_s,
            )
        else:
            obs = np.array([
                state.displacement_m / TRAINING.displacement_scale_m,
                state.velocity_m_s / TRAINING.velocity_scale_m_s,
                previous_action,
            ], dtype=np.float32)

            action, _ = sac.predict(obs, deterministic=True)
            previous_action = float(np.clip(
                np.asarray(action).reshape(-1)[0], -1.0, 1.0
            ))
            force = previous_action * PLANT.max_control_force_n

        next_state = plant.step(
            disturbance(i * PLANT.dt_s), force
        )
        x[i] = next_state.displacement_m * 1e6  # micrometres
        u[i] = force

    steady_x = x[steady_start:]
    steady_u = u[steady_start:]

    return {
        "rms_vibration_um": float(np.sqrt(np.mean(x ** 2))),
        "steady_rms_vibration_um": float(np.sqrt(np.mean(steady_x ** 2))),
        "peak_vibration_um": float(np.max(np.abs(x))),
        "rms_control_force_n": float(np.sqrt(np.mean(steady_u ** 2))),
        "peak_control_force_n": float(np.max(np.abs(steady_u))),
        "saturation_percent": float(100 * np.mean(
            np.abs(steady_u) >= 0.99 * PLANT.max_control_force_n
        )),
    }


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {MODEL_PATH}. Train SAC first or correct MODEL_PATH."
        )

    if not PID_GAINS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {PID_GAINS_PATH}. Run tune_pid.py first."
        )

    with PID_GAINS_PATH.open("r", encoding="utf-8") as f:
        pid_gains = json.load(f)

    print("Using tuned PID gains:", pid_gains)

    OUT.mkdir(exist_ok=True)
    sac = SAC.load(str(MODEL_PATH))
    conditions = make_conditions()
    rows = []

    total = len(conditions) * 3
    done = 0
    print(f"Testing {len(conditions)} conditions; {total} simulations total.")

    for cid, condition in enumerate(conditions, start=1):
        # All controllers get exactly the same amplitude, frequency and phase.
        for name in ["No control", "PID", "SAC"]:
            metrics = simulate(
                name,
                condition,
                sac=sac if name == "SAC" else None,
                pid_gains=pid_gains,
            )
            rows.append({
                "condition_id": cid,
                **condition,
                "controller": name,
                **metrics,
            })
            done += 1

        if cid % 6 == 0 or cid == len(conditions):
            print(f"Completed {done}/{total} simulations")

    detail = pd.DataFrame(rows)

    # Compare each controller against No control for the SAME condition.
    baseline = detail[detail.controller == "No control"][
        ["condition_id", "steady_rms_vibration_um"]
    ].rename(columns={
        "steady_rms_vibration_um": "no_control_steady_rms_um"
    })

    detail = detail.merge(baseline, on="condition_id", how="left")
    detail["reduction_percent"] = 100 * (
        detail.no_control_steady_rms_um - detail.steady_rms_vibration_um
    ) / detail.no_control_steady_rms_um.replace(0, np.nan)

    detail.to_csv(OUT / "robust_evaluation_by_condition.csv", index=False)

    # Summarise all conditions and the in-range / out-of-range frequency groups.
    summaries = []
    groups = ["all", "within_training_range", "outside_training_range"]

    for group in groups:
        part = detail if group == "all" else detail[
            detail.frequency_group == group
        ]

        for name in ["No control", "PID", "SAC"]:
            c = part[part.controller == name]
            reductions = c.reduction_percent.dropna()

            summaries.append({
                "frequency_group": group,
                "controller": name,
                "number_of_conditions": len(c),
                "mean_steady_rms_um": c.steady_rms_vibration_um.mean(),
                "median_steady_rms_um": c.steady_rms_vibration_um.median(),
                "mean_reduction_percent": reductions.mean(),
                "median_reduction_percent": reductions.median(),
                "percent_conditions_better_than_no_control":
                    100 * (reductions > 0).mean(),
                "mean_control_force_n": c.rms_control_force_n.mean(),
                "mean_saturation_percent": c.saturation_percent.mean(),
            })

    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "robust_evaluation_summary.csv", index=False)

    # Plot average steady RMS vibration by frequency group.
    plot_groups = ["within_training_range", "outside_training_range"]
    labels = ["Within training range", "Outside training range"]
    controllers = ["No control", "PID", "SAC"]
    xloc = np.arange(2)
    width = 0.24

    fig, ax = plt.subplots(figsize=(10, 5))
    for j, name in enumerate(controllers):
        values = []
        for group in plot_groups:
            row = summary[
                (summary.frequency_group == group)
                & (summary.controller == name)
            ]
            values.append(row.mean_steady_rms_um.iloc[0])
        ax.bar(xloc + (j - 1) * width, values, width, label=name)

    ax.set_xticks(xloc)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean steady RMS vibration (micrometres)")
    ax.set_title("Average vibration: lower is better")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "robust_evaluation_rms.png", dpi=160)
    plt.close(fig)

    # Plot vibration reduction relative to No control.
    fig, ax = plt.subplots(figsize=(10, 5))
    for j, name in enumerate(controllers):
        values = []
        for group in plot_groups:
            row = summary[
                (summary.frequency_group == group)
                & (summary.controller == name)
            ]
            values.append(row.mean_reduction_percent.iloc[0])
        ax.bar(xloc + (j - 1) * width, values, width, label=name)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(xloc)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean vibration reduction vs No control (%)")
    ax.set_title("Positive is better; negative means vibration increased")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "robust_evaluation_reduction.png", dpi=160)
    plt.close(fig)

    columns = [
        "frequency_group", "controller", "number_of_conditions",
        "mean_steady_rms_um", "mean_reduction_percent",
        "median_reduction_percent",
        "percent_conditions_better_than_no_control",
        "mean_control_force_n", "mean_saturation_percent",
    ]
    print("\nSUMMARY:")
    print(summary[columns].round(4).to_string(index=False))
    print("\nSaved results:")
    print("results\\robust_evaluation_by_condition.csv")
    print("results\\robust_evaluation_summary.csv")
    print("results\\robust_evaluation_rms.png")
    print("results\\robust_evaluation_reduction.png")


if __name__ == "__main__":
    main()

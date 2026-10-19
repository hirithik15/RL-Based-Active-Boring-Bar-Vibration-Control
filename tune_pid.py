
"""Tune PID on training conditions and report separate validation results."""
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd

from config import PLANT
from controllers import PIDController
from models import SingleModePlant, SinusoidalCuttingForce
from config import DisturbanceConfig

# Search grid: extend it later if the best gains lie at a boundary.
KP_VALUES = [0, 2500, 5000, 8000, 12000, 18000, 25000, 35000, 50000]
KD_VALUES = [0, 1, 2, 3, 5, 7, 10, 14, 20]
DURATION_S = 0.2
STEADY_START_FRACTION = 0.2

# Force penalty is intentionally small; saturation is penalized strongly.
FORCE_PENALTY_WEIGHT = 0.02
SATURATION_PENALTY_WEIGHT = 10.0


def make_conditions(validation=False):
    """Training conditions select gains; validation conditions do not."""
    if not validation:
        amplitudes = (0.5, 1.0, 1.5)
        frequencies = (150.0, 174.0, 200.0)
        phases = (0.0, math.pi / 2)
    else:
        # Out-of-range validation is for reporting only.
        amplitudes = (0.75, 1.25)
        frequencies = (140.0, 210.0, 220.0)
        phases = (0.0, math.pi / 2, math.pi, 3 * math.pi / 2)

    return [
        (a, f, p)
        for a in amplitudes
        for f in frequencies
        for p in phases
    ]


def run_one(kp, kd, condition):
    amplitude, frequency, phase = condition

    def simulate(controller_enabled):
        plant = SingleModePlant(PLANT)
        plant.reset(0.0, 0.0)
        disturbance = SinusoidalCuttingForce(
            DisturbanceConfig(
                amplitude_n=amplitude,
                frequency_hz=frequency,
                phase_rad=phase,
            )
        )
        pid = PIDController(
            kp=kp if controller_enabled else 0.0,
            ki=0.0,
            kd=kd if controller_enabled else 0.0,
            plant_config=PLANT,
        )
        pid.reset()

        n = round(DURATION_S / PLANT.dt_s)
        start = int(n * STEADY_START_FRACTION)
        x_um = np.zeros(n)
        u_n = np.zeros(n)

        for i in range(n):
            state = plant.state
            force = (
                pid.act(state.displacement_m, state.velocity_m_s, PLANT.dt_s)
                if controller_enabled else 0.0
            )
            next_state = plant.step(
                disturbance(i * PLANT.dt_s), force
            )
            x_um[i] = next_state.displacement_m * 1e6
            u_n[i] = force

        x_steady = x_um[start:]
        u_steady = u_n[start:]
        return {
            "rms_um": float(np.sqrt(np.mean(x_steady**2))),
            "rms_force_n": float(np.sqrt(np.mean(u_steady**2))),
            "saturation_fraction": float(np.mean(
                np.abs(u_steady) >= 0.99 * PLANT.max_control_force_n
            )),
        }

    baseline = simulate(False)
    controlled = simulate(True)
    reduction = (
        100.0 * (baseline["rms_um"] - controlled["rms_um"])
        / max(baseline["rms_um"], 1e-12)
    )

    # Relative RMS is 1.0 for no control; values below 1.0 are better.
    relative_rms = controlled["rms_um"] / max(baseline["rms_um"], 1e-12)
    normalized_force = controlled["rms_force_n"] / PLANT.max_control_force_n
    score = (
        relative_rms
        + FORCE_PENALTY_WEIGHT * normalized_force**2
        + SATURATION_PENALTY_WEIGHT * controlled["saturation_fraction"]
    )

    return {
        "kp": kp,
        "ki": 0.0,
        "kd": kd,
        "rms_um": controlled["rms_um"],
        "no_control_rms_um": baseline["rms_um"],
        "reduction_percent": reduction,
        "rms_force_n": controlled["rms_force_n"],
        "saturation_fraction": controlled["saturation_fraction"],
        "score": score,
    }


def evaluate_grid(conditions):
    rows = []
    total = len(KP_VALUES) * len(KD_VALUES)
    for index, (kp, kd) in enumerate(
        ((kp, kd) for kp in KP_VALUES for kd in KD_VALUES), start=1
    ):
        rows.extend(run_one(kp, kd, c) for c in conditions)
        if index % 9 == 0 or index == total:
            print(f"Tested {index}/{total} gain combinations.")
    return pd.DataFrame(rows)


def main():
    out = Path("results")
    out.mkdir(parents=True, exist_ok=True)

    train_conditions = make_conditions(validation=False)
    validation_conditions = make_conditions(validation=True)

    print("Tuning on training-range conditions...")
    train_df = evaluate_grid(train_conditions)

    print("\nUnique gain combinations:")
    print(train_df[["kp", "ki", "kd"]].drop_duplicates().to_string(index=False))

    print("\nNumber of unique combinations:",
        train_df[["kp", "ki", "kd"]].drop_duplicates().shape[0])

    print("\nBest 20 rows:")
        
    print(train_df.sort_values("score").head(20).to_string(index=False))
    # Average each metric across training conditions for each gain pair.
    summary = train_df.groupby(["kp", "ki", "kd"], as_index=False).agg(
        mean_rms_um=("rms_um", "mean"),
        mean_no_control_rms_um=("no_control_rms_um", "mean"),
        mean_reduction_percent=("reduction_percent", "mean"),
        mean_rms_force_n=("rms_force_n", "mean"),
        mean_saturation_fraction=("saturation_fraction", "mean"),
        score=("score", "mean"),
    ).sort_values("score")

    summary.to_csv(out / "pid_tuning_results.csv", index=False)
    best = summary.iloc[0]
    gains = {
        "kp": float(best.kp),
        "ki": float(best.ki),
        "kd": float(best.kd),
        "selection_metric": "mean normalized RMS + effort penalty + saturation penalty",
        "training_condition_count": len(train_conditions),
    }
    with (out / "tuned_pid.json").open("w", encoding="utf-8") as f:
        json.dump(gains, f, indent=2)

    print("\nBest PID candidates (training conditions only):")
    print(summary.head(10).round(6).to_string(index=False))
    print("\nSelected gains:")
    print(json.dumps(gains, indent=2))

    # Report validation only for the selected gain pair; do not re-select gains.
    print("\nEvaluating selected gains on out-of-range validation conditions...")
    val_rows = [
        run_one(gains["kp"], gains["kd"], c)
        for c in validation_conditions
    ]
    val_df = pd.DataFrame(val_rows)
    val_df.to_csv(out / "pid_validation_conditions.csv", index=False)
    print(val_df[[
        "rms_um", "no_control_rms_um", "reduction_percent",
        "rms_force_n", "saturation_fraction"
    ]].mean().round(6).to_string())
    print("\nSaved:")
    print("results\\pid_tuning_results.csv")
    print("results\\tuned_pid.json")
    print("results\\pid_validation_conditions.csv")


if __name__ == "__main__":
    main()

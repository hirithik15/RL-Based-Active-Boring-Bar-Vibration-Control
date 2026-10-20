"""Tune PID against the current machining-aware simulation service.

Run from the project root:
    python tune_pid.py

This tuner calls deployment_service.run_simulation, so it evaluates the same
MachiningConfig / stochastic-regenerative force model as the dashboard. It
does not tune against the old sinusoidal test disturbance.

The search is deliberately broad and logarithmic. Gains are selected using
training conditions only; separate validation conditions are reported after
selection. Runtime depends on the simulation timestep and grid size.
"""

from pathlib import Path
import json
import itertools

import numpy as np
import pandas as pd

from controllers import PIDController
from deployment_service import run_simulation


OUTPUT_DIR = Path("results")
DURATION_S = 0.2

# Broad force-domain gain ranges. These are search bounds, not guaranteed
# stable gains. Log spacing explores orders of magnitude without over-weighting
# the largest values. Zero is included to test pure P / pure D baselines.
KP_VALUES = np.concatenate(([0.0], np.geomspace(1e3, 1e7, 13)))
KD_VALUES = np.concatenate(([0.0], np.geomspace(0.1, 1e3, 13)))
KI_VALUES = [0.0]  # First tune PD; tune integral action only if justified.

FORCE_PENALTY_WEIGHT = 0.02
SATURATION_PENALTY_WEIGHT = 2.0
PEAK_PENALTY_WEIGHT = 0.10

BASE_CONDITION = {
    "spindle_rpm": 9000.0,
    "feed_mm_rev": 0.10,
    "specific_cutting_force_n_mm2": 1800.0,
    "force_fluctuation_fraction": 0.20,
    "random_force_fraction": 0.08,
    "modal_participation": 0.02,
    "transient_rate_hz": 3.0,
    "transient_fraction": 0.50,
    "phase_rad": 0.0,
    "duration_s": DURATION_S,
}

def make_conditions(validation=False):
    """Create representative machining conditions with different seeds."""
    if not validation:
        rpm_values = (7000.0, 9000.0, 11000.0)
        feed_values = (0.06, 0.10, 0.14)
        depth_values = (0.25, 0.50)
        seeds = (101, 202)
    else:
        # Held-out combinations: don't use these to select gains.
        rpm_values = (8000.0, 10500.0)
        feed_values = (0.08, 0.12)
        depth_values = (0.35, 0.65)
        seeds = (303, 404, 505)

    conditions = []
    for rpm, feed, depth, seed in itertools.product(
        rpm_values, feed_values, depth_values, seeds
    ):
        condition = dict(BASE_CONDITION)
        condition.update({
            "spindle_rpm": rpm,
            "feed_mm_rev": feed,
            "depth_of_cut_mm": depth,
            "seed": seed,
        })
        conditions.append(condition)
    return conditions


def _metric(metrics, *keys, default=np.nan):
    for key in keys:
        if key in metrics:
            try:
                value = float(metrics[key])
                if np.isfinite(value):
                    return value
            except (TypeError, ValueError):
                pass
    return default


def evaluate_one(kp, ki, kd, condition, baseline_metrics=None):
    """Evaluate one candidate against the same no-control condition/seed."""
    pid_result = run_simulation(
        controller_name="PID",
        **condition,
        pid_kp=float(kp),
        pid_ki=float(ki),
        pid_kd=float(kd),
    )
    metrics = pid_result["metrics"]

    if baseline_metrics is None:
        baseline_result = run_simulation(
            controller_name="No control",
            **condition,
        )
        baseline_metrics = baseline_result["metrics"]

    rms = _metric(metrics, "steady_rms_vibration_um", "rms_vibration_um")
    baseline_rms = _metric(
        baseline_metrics, "steady_rms_vibration_um", "rms_vibration_um"
    )
    peak = _metric(metrics, "peak_vibration_um")
    baseline_peak = _metric(baseline_metrics, "peak_vibration_um")
    rms_force = _metric(metrics, "rms_control_force_n", default=0.0)
    force_limit = float(PLANT_MAX_FORCE_N)
    saturation = _metric(metrics, "actuator_saturation_fraction", default=0.0)

    relative_rms = rms / max(baseline_rms, 1e-12)
    relative_peak = peak / max(baseline_peak, 1e-12)
    normalized_force = rms_force / max(force_limit, 1e-12)
    score = (
        relative_rms
        + PEAK_PENALTY_WEIGHT * relative_peak
        + FORCE_PENALTY_WEIGHT * normalized_force**2
        + SATURATION_PENALTY_WEIGHT * saturation
    )

    return {
        "kp": float(kp),
        "ki": float(ki),
        "kd": float(kd),
        "rms_um": rms,
        "baseline_rms_um": baseline_rms,
        "rms_reduction_percent": 100.0 * (1.0 - relative_rms),
        "peak_um": peak,
        "baseline_peak_um": baseline_peak,
        "rms_control_force_n": rms_force,
        "saturation_fraction": saturation,
        "score": float(score),
        "seed": condition["seed"],
        "spindle_rpm": condition["spindle_rpm"],
        "feed_mm_rev": condition["feed_mm_rev"],
        "depth_of_cut_mm": condition["depth_of_cut_mm"],
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_conditions = make_conditions(validation=False)
    validation_conditions = make_conditions(validation=True)

    # Cache baseline results once per training condition.
    print(f"Computing {len(train_conditions)} training baselines...")
    baselines = []
    for i, condition in enumerate(train_conditions, start=1):
        result = run_simulation(controller_name="No control", **condition)
        baselines.append(result["metrics"])
        if i % 10 == 0 or i == len(train_conditions):
            print(f"  Baselines: {i}/{len(train_conditions)}")

    candidates = list(itertools.product(KP_VALUES, KI_VALUES, KD_VALUES))
    total = len(candidates) * len(train_conditions)
    print(f"\nTesting {len(candidates)} gain combinations over "
          f"{len(train_conditions)} conditions ({total} simulations)...")

    rows = []
    for idx, (kp, ki, kd) in enumerate(candidates, start=1):
        for condition, baseline_metrics in zip(train_conditions, baselines):
            try:
                rows.append(evaluate_one(
                    kp, ki, kd, condition, baseline_metrics=baseline_metrics
                ))
            except (ValueError, RuntimeError, FloatingPointError) as exc:
                # Failed/unstable candidates are kept in the report but ranked
                # behind all finite candidates.
                rows.append({
                    "kp": float(kp), "ki": float(ki), "kd": float(kd),
                    "score": float("inf"), "error": str(exc),
                    "seed": condition["seed"],
                    "spindle_rpm": condition["spindle_rpm"],
                    "feed_mm_rev": condition["feed_mm_rev"],
                    "depth_of_cut_mm": condition["depth_of_cut_mm"],
                })
        if idx % 10 == 0 or idx == len(candidates):
            print(f"  Gain combinations: {idx}/{len(candidates)}")

    raw = pd.DataFrame(rows)
    raw.to_csv(OUTPUT_DIR / "pid_tuning_trials.csv", index=False)

    valid = raw[np.isfinite(pd.to_numeric(raw["score"], errors="coerce"))].copy()
    if valid.empty:
        raise RuntimeError(
            "No valid PID candidates. Check the API/model code and simulation logs."
        )

    summary = valid.groupby(["kp", "ki", "kd"], as_index=False).agg(
        mean_rms_um=("rms_um", "mean"),
        mean_baseline_rms_um=("baseline_rms_um", "mean"),
        mean_rms_reduction_percent=("rms_reduction_percent", "mean"),
        mean_peak_um=("peak_um", "mean"),
        mean_rms_control_force_n=("rms_control_force_n", "mean"),
        mean_saturation_fraction=("saturation_fraction", "mean"),
        mean_score=("score", "mean"),
        worst_score=("score", "max"),
    )

    # Favor low average score while also discouraging a poor worst-case result.
    summary["robust_score"] = summary["mean_score"] + 0.25 * summary["worst_score"]
    summary = summary.sort_values(["robust_score", "mean_score"])
    summary.to_csv(OUTPUT_DIR / "pid_tuning_results.csv", index=False)

    best = summary.iloc[0]
    gains = {
        "kp": float(best["kp"]),
        "ki": float(best["ki"]),
        "kd": float(best["kd"]),
        "selection_metric": (
            "mean normalized steady RMS + peak displacement penalty + "
            "control-effort penalty + saturation penalty; with worst-case weighting"
        ),
        "training_condition_count": len(train_conditions),
        "search_bounds": {
            "kp": [float(np.min(KP_VALUES)), float(np.max(KP_VALUES))],
            "ki": [float(min(KI_VALUES)), float(max(KI_VALUES))],
            "kd": [float(np.min(KD_VALUES)), float(np.max(KD_VALUES))],
        },
        "validation_status": "not yet evaluated",
    }

    print("\nTop 15 candidates:")
    print(summary.head(15).round(5).to_string(index=False))
    print("\nSelected gains from training conditions:")
    print(json.dumps(gains, indent=2))

    # Validate selected gains on held-out machining conditions/seeds.
    print(f"\nValidating selected gains on {len(validation_conditions)} held-out conditions...")
    validation_rows = []
    for i, condition in enumerate(validation_conditions, start=1):
        try:
            validation_rows.append(evaluate_one(
                gains["kp"], gains["ki"], gains["kd"], condition
            ))
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            validation_rows.append({
                "kp": gains["kp"], "ki": gains["ki"], "kd": gains["kd"],
                "score": float("inf"), "error": str(exc),
                "seed": condition["seed"],
            })
        if i % 5 == 0 or i == len(validation_conditions):
            print(f"  Validation: {i}/{len(validation_conditions)}")

    validation_df = pd.DataFrame(validation_rows)
    validation_df.to_csv(OUTPUT_DIR / "pid_validation_conditions.csv", index=False)

    finite_val = validation_df[
        np.isfinite(pd.to_numeric(validation_df.get("score"), errors="coerce"))
    ]
    if not finite_val.empty:
        gains["validation_status"] = "evaluated"
        gains["validation_mean_rms_reduction_percent"] = float(
            finite_val["rms_reduction_percent"].mean()
        )
        gains["validation_mean_saturation_fraction"] = float(
            finite_val["saturation_fraction"].mean()
        )
        gains["validation_mean_score"] = float(finite_val["score"].mean())
    else:
        gains["validation_status"] = "failed; inspect pid_validation_conditions.csv"

    with (OUTPUT_DIR / "tuned_pid.json").open("w", encoding="utf-8") as handle:
        json.dump(gains, handle, indent=2)

    print("\nValidation summary:")
    if not finite_val.empty:
        print(f"Mean RMS reduction: {gains['validation_mean_rms_reduction_percent']:.3f}%")
        print(f"Mean saturation fraction: {gains['validation_mean_saturation_fraction']:.4f}")
        print(f"Mean score: {gains['validation_mean_score']:.5f}")
    else:
        print("No valid validation rows; inspect the CSV for errors.")

    print("\nSaved:")
    print(f"  {OUTPUT_DIR / 'pid_tuning_trials.csv'}")
    print(f"  {OUTPUT_DIR / 'pid_tuning_results.csv'}")
    print(f"  {OUTPUT_DIR / 'pid_validation_conditions.csv'}")
    print(f"  {OUTPUT_DIR / 'tuned_pid.json'}")
    print("\nRestart FastAPI after tuning so the service loads the saved gains.")


# Read plant force limit from the project's config without hard-coding units.
try:
    from config import PLANT
    PLANT_MAX_FORCE_N = float(PLANT.max_control_force_n)
except Exception:
    PLANT_MAX_FORCE_N = 1.0


if __name__ == "__main__":
    main()

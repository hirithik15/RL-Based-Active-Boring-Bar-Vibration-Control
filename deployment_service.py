
"""Shared simulation service for the vibration-control deployment demo.

This service runs the existing single-mode simulation. It does not connect
to physical sensors, actuators, or a live boring-bar machine.
"""
from functools import lru_cache

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

from config import PLANT, TRAINING, EVALUATION, DisturbanceConfig
from experiments import simulate_controller, calculate_metrics


DEFAULT_MODEL_PATH = "results/sac_vibration_model_v2.zip"
ALLOWED_CONTROLLERS = ("No control", "PID", "SAC")


@lru_cache(maxsize=2)
def _load_sac_model(model_path):
    """Load and cache a trained SAC model; never train during inference."""
    return SAC.load(model_path)


def get_model_status(model_path=DEFAULT_MODEL_PATH):
    """Report whether the trained model file is available."""
    from pathlib import Path

    path = Path(model_path)
    return {
        "model_path": str(path),
        "model_exists": path.is_file(),
        "message": (
            "Trained SAC model is available."
            if path.is_file()
            else "Trained SAC model not found. Train or place the model at this path."
        ),
    }


def run_simulation(
    controller_name="SAC",
    amplitude_n=EVALUATION.nominal_amplitude_n,
    frequency_hz=EVALUATION.nominal_frequency_hz,
    phase_rad=0.0,
    duration_s=EVALUATION.duration_s,
    model_path=DEFAULT_MODEL_PATH,
):
    """Run one controller and return metrics plus its time-series data.

    Parameters are validated here so the API and dashboard share the same
    simulation behavior.
    """
    if controller_name not in ALLOWED_CONTROLLERS:
        raise ValueError(
            f"controller_name must be one of {ALLOWED_CONTROLLERS}."
        )

    numeric_values = {
        "amplitude_n": amplitude_n,
        "frequency_hz": frequency_hz,
        "phase_rad": phase_rad,
        "duration_s": duration_s,
    }
    for name, value in numeric_values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite number.")
        if not np.isfinite(value):
            raise ValueError(f"{name} must be a finite number.")

    if amplitude_n < 0:
        raise ValueError("amplitude_n must be non-negative.")
    if frequency_hz < 0:
        raise ValueError("frequency_hz must be non-negative.")
    if duration_s <= 0:
        raise ValueError("duration_s must be positive.")
    if duration_s > 2.0:
        raise ValueError("duration_s must not exceed 2 seconds.")

    # Bound simulation size to prevent accidental resource exhaustion.
    step_count = int(round(duration_s / PLANT.dt_s))
    if step_count < 1 or step_count > 20_000:
        raise ValueError("Requested duration produces an invalid step count.")

    disturbance = DisturbanceConfig(
        amplitude_n=float(amplitude_n),
        frequency_hz=float(frequency_hz),
        phase_rad=float(phase_rad),
    )

    model = None
    if controller_name == "SAC":
        status = get_model_status(model_path)
        if not status["model_exists"]:
            raise FileNotFoundError(status["message"])
        model = _load_sac_model(str(model_path))

    data = simulate_controller(
        controller_name=controller_name,
        disturbance_config=disturbance,
        duration_s=float(duration_s),
        model=model,
    )

    metrics = calculate_metrics(data)

    # Convert the DataFrame to ordinary dictionaries for JSON/API use.
    records = data.replace([np.inf, -np.inf], np.nan).to_dict(
        orient="records"
    )
    if any(
        pd.isna(value)
        for record in records
        for value in record.values()
    ):
        raise RuntimeError("Simulation produced invalid numeric output.")

    return {
        "simulation_only": True,
        "model_type": "single-mode harmonic-disturbance surrogate",
        "controller": controller_name,
        "inputs": {
            "amplitude_n": float(amplitude_n),
            "frequency_hz": float(frequency_hz),
            "phase_rad": float(phase_rad),
            "duration_s": float(duration_s),
        },
        "training_frequency_range_hz": [
            TRAINING.force_frequency_min_hz,
            TRAINING.force_frequency_max_hz,
        ],
        "frequency_outside_training_range": not (
            TRAINING.force_frequency_min_hz
            <= frequency_hz
            <= TRAINING.force_frequency_max_hz
        ),
        "metrics": metrics,
        "timeseries": records,
    }


def compare_controllers(
    amplitude_n=EVALUATION.nominal_amplitude_n,
    frequency_hz=EVALUATION.nominal_frequency_hz,
    phase_rad=0.0,
    duration_s=EVALUATION.duration_s,
    model_path=DEFAULT_MODEL_PATH,
):
    """Run all three controllers under the same disturbance condition."""
    results = []
    for controller_name in ALLOWED_CONTROLLERS:
        result = run_simulation(
            controller_name=controller_name,
            amplitude_n=amplitude_n,
            frequency_hz=frequency_hz,
            phase_rad=phase_rad,
            duration_s=duration_s,
            model_path=model_path,
        )
        results.append({
            "controller": controller_name,
            **result["metrics"],
        })

    return results

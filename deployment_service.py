"""Simulation service for the machining-aware vibration-control dashboard."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

from config import PLANT, MachiningConfig
from experiments import simulate_controller, calculate_metrics
from controllers import PIDController


DEFAULT_MODEL_PATH = "results/sac_regenerative_v1.zip"
ALLOWED_CONTROLLERS = ("No control", "Active damping", "PID", "SAC")


@lru_cache(maxsize=2)
def _load_sac_model(model_path):
    return SAC.load(model_path)


def get_model_status(model_path=DEFAULT_MODEL_PATH):
    path = Path(model_path)
    return {
        "model_path": str(path),
        "model_exists": path.is_file(),
        "message": (
            "SAC model is available."
            if path.is_file()
            else "SAC model not found. Train the updated model first."
        ),
    }


def _validate_number(name, value, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number.")
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")


def _make_machining_config(
    spindle_rpm,
    feed_mm_rev,
    depth_of_cut_mm,
    specific_cutting_force_n_mm2,
    force_fluctuation_fraction,
    random_force_fraction,
    modal_participation,
    transient_rate_hz,
    transient_fraction,
    phase_rad,
):
    return MachiningConfig(
        spindle_rpm=float(spindle_rpm),
        feed_mm_rev=float(feed_mm_rev),
        depth_of_cut_mm=float(depth_of_cut_mm),
        specific_cutting_force_n_mm2=float(specific_cutting_force_n_mm2),
        force_fluctuation_fraction=float(force_fluctuation_fraction),
        random_force_fraction=float(random_force_fraction),
        modal_participation=float(modal_participation),
        transient_rate_hz=float(transient_rate_hz),
        transient_fraction=float(transient_fraction),
        transient_decay_s=0.003,
        regenerative_feedback_sign=1.0,
        phase_rad=float(phase_rad),
    )


def run_simulation(
    controller_name="SAC",
    spindle_rpm=9000.0,
    feed_mm_rev=0.10,
    depth_of_cut_mm=0.50,
    specific_cutting_force_n_mm2=1800.0,
    force_fluctuation_fraction=0.20,
    random_force_fraction=0.08,
    modal_participation=0.02,
    transient_rate_hz=3.0,
    transient_fraction=0.50,
    phase_rad=0.0,
    duration_s=0.2,
    seed=12345,
    model_path=DEFAULT_MODEL_PATH,
    pid_kp=None,
    pid_ki=None,
    pid_kd=None,
):
    """Run one controller with the hybrid stochastic/regenerative model.

    Optional PID gains apply only to this call. If omitted, PIDController
    loads results/tuned_pid.json (or its fallback gains).
    """
    if controller_name not in ALLOWED_CONTROLLERS:
        raise ValueError(f"controller_name must be one of {ALLOWED_CONTROLLERS}.")

    numeric_inputs = {
        "spindle_rpm": (spindle_rpm, 1, 120000),
        "feed_mm_rev": (feed_mm_rev, 0.001, 5),
        "depth_of_cut_mm": (depth_of_cut_mm, 0.001, 20),
        "specific_cutting_force_n_mm2": (specific_cutting_force_n_mm2, 1, 10000),
        "force_fluctuation_fraction": (force_fluctuation_fraction, 0, 2),
        "random_force_fraction": (random_force_fraction, 0, 2),
        "modal_participation": (modal_participation, 0.0001, 1),
        "transient_rate_hz": (transient_rate_hz, 0, 100),
        "transient_fraction": (transient_fraction, 0, 5),
        "phase_rad": (phase_rad, -100, 100),
        "duration_s": (duration_s, 0.001, 2.0),
    }
    for name, (value, minimum, maximum) in numeric_inputs.items():
        _validate_number(name, value, minimum, maximum)

    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")

    steps = int(round(duration_s / PLANT.dt_s))
    if steps < 1 or steps > 20000:
        raise ValueError("Invalid simulation step count.")

    machining = _make_machining_config(
        spindle_rpm,
        feed_mm_rev,
        depth_of_cut_mm,
        specific_cutting_force_n_mm2,
        force_fluctuation_fraction,
        random_force_fraction,
        modal_participation,
        transient_rate_hz,
        transient_fraction,
        phase_rad,
    )

    model = None
    if controller_name == "SAC":
        status = get_model_status(model_path)
        if not status["model_exists"]:
            raise FileNotFoundError(status["message"])
        model = _load_sac_model(str(model_path))

    # For PID, passing explicit gains avoids shared mutable defaults and
    # makes this service call deterministic even during parallel requests.
    if controller_name == "PID" and any(
        gain is not None for gain in (pid_kp, pid_ki, pid_kd)
    ):
        defaults = PIDController.DEFAULT_GAINS
        try:
            PIDController.DEFAULT_GAINS = (
                float(defaults[0] if pid_kp is None else pid_kp),
                float(defaults[1] if pid_ki is None else pid_ki),
                float(defaults[2] if pid_kd is None else pid_kd),
            )
            for gain_name, gain in (
                ("pid_kp", PIDController.DEFAULT_GAINS[0]),
                ("pid_ki", PIDController.DEFAULT_GAINS[1]),
                ("pid_kd", PIDController.DEFAULT_GAINS[2]),
            ):
                _validate_number(gain_name, gain, 0, 1e12)
            data = simulate_controller(
                controller_name=controller_name,
                machining_config=machining,
                duration_s=float(duration_s),
                model=model,
                seed=int(seed),
            )
        finally:
            PIDController.DEFAULT_GAINS = defaults
    else:
        data = simulate_controller(
            controller_name=controller_name,
            machining_config=machining,
            duration_s=float(duration_s),
            model=model,
            seed=int(seed),
        )

    metrics = calculate_metrics(data)
    records = data.replace([np.inf, -np.inf], np.nan).to_dict(orient="records")
    if any(pd.isna(value) for record in records for value in record.values()):
        raise RuntimeError("Simulation produced invalid numeric output.")

    return {
        "simulation_only": True,
        "model_type": "stochastic cutting force with regenerative feedback",
        "controller": controller_name,
        "seed": int(seed),
        "inputs": {
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
        },
        "pid_gains": (
            {
                "kp": float(pid_kp if pid_kp is not None else PIDController.DEFAULT_GAINS[0]),
                "ki": float(pid_ki if pid_ki is not None else PIDController.DEFAULT_GAINS[1]),
                "kd": float(pid_kd if pid_kd is not None else PIDController.DEFAULT_GAINS[2]),
            }
            if controller_name == "PID" else None
        ),
        "metrics": metrics,
        "timeseries": records,
    }


def compare_controllers(
    spindle_rpm=9000.0,
    feed_mm_rev=0.10,
    depth_of_cut_mm=0.50,
    duration_s=0.2,
    seed=12345,
    model_path=DEFAULT_MODEL_PATH,
    specific_cutting_force_n_mm2=1800.0,
    force_fluctuation_fraction=0.20,
    random_force_fraction=0.08,
    modal_participation=0.02,
    transient_rate_hz=3.0,
    transient_fraction=0.50,
    phase_rad=0.0,
):
    """Compare controllers under the same full machining condition."""
    common = dict(
        spindle_rpm=spindle_rpm,
        feed_mm_rev=feed_mm_rev,
        depth_of_cut_mm=depth_of_cut_mm,
        specific_cutting_force_n_mm2=specific_cutting_force_n_mm2,
        force_fluctuation_fraction=force_fluctuation_fraction,
        random_force_fraction=random_force_fraction,
        modal_participation=modal_participation,
        transient_rate_hz=transient_rate_hz,
        transient_fraction=transient_fraction,
        phase_rad=phase_rad,
        duration_s=duration_s,
        seed=seed,
        model_path=model_path,
    )
    results = []
    for name in ALLOWED_CONTROLLERS:
        result = run_simulation(controller_name=name, **common)
        results.append({"controller": name, **result["metrics"]})
    return results

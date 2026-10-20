"""Controllers for the single-mode boring-bar vibration model.

PID gains are force-domain simulation gains:
    F_control = -kp*x - kd*xdot - ki*integral(x dt)

If results/tuned_pid.json exists, its gains become the default PID gains.
Otherwise, conservative starter values are used. These values must be tuned
and validated for the current plant and machining-force model.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import ClassVar, Optional, Tuple

import numpy as np

from config import PLANT, PlantConfig


DEFAULT_PID_GAINS: Tuple[float, float, float] = (50000.0, 0.0, 20.0)
TUNED_PID_PATH = Path("results/tuned_pid.json")


def _read_saved_pid_gains():
    """Read tuned gains if a valid tuning file exists; otherwise use fallback."""
    if not TUNED_PID_PATH.is_file():
        return DEFAULT_PID_GAINS

    try:
        with TUNED_PID_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        kp = float(data["kp"])
        ki = float(data.get("ki", 0.0))
        kd = float(data["kd"])
        if not all(np.isfinite(v) and v >= 0 for v in (kp, ki, kd)):
            return DEFAULT_PID_GAINS
        return kp, ki, kd
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return DEFAULT_PID_GAINS


class NoControl:
    """Baseline controller that applies no active force."""

    def reset(self):
        pass

    def act(self, displacement_m, velocity_m_s, dt_s):
        return 0.0


@dataclass
class ActiveDampingController:
    """Velocity-feedback damping in the force-command domain."""

    velocity_gain: float = 20.0
    plant_config: PlantConfig = PLANT

    def __post_init__(self):
        if not np.isfinite(self.velocity_gain) or self.velocity_gain < 0:
            raise ValueError("velocity_gain must be finite and non-negative.")

    def reset(self):
        pass

    def act(self, displacement_m, velocity_m_s, dt_s):
        if dt_s <= 0:
            raise ValueError("dt_s must be positive.")
        force_n = -self.velocity_gain * float(velocity_m_s)
        limit_n = self.plant_config.max_control_force_n
        return float(np.clip(force_n, -limit_n, limit_n))


@dataclass
class PIDController:
    """PID baseline with force saturation and conditional anti-windup.

    Set kp/ki/kd explicitly to override the defaults. If omitted, the
    controller loads results/tuned_pid.json when available.
    """

    # Class-level override used by tune_pid.py while testing each candidate.
    DEFAULT_GAINS: ClassVar[Tuple[float, float, float]] = _read_saved_pid_gains()

    kp: Optional[float] = None
    ki: Optional[float] = None
    kd: Optional[float] = None
    plant_config: PlantConfig = PLANT
    integral_limit_m_s: float = 1e-3
    integral_error: float = 0.0

    def __post_init__(self):
        fallback_kp, fallback_ki, fallback_kd = type(self).DEFAULT_GAINS
        if self.kp is None:
            self.kp = fallback_kp
        if self.ki is None:
            self.ki = fallback_ki
        if self.kd is None:
            self.kd = fallback_kd

        self.kp = float(self.kp)
        self.ki = float(self.ki)
        self.kd = float(self.kd)

        if not all(np.isfinite(v) and v >= 0 for v in (self.kp, self.ki, self.kd)):
            raise ValueError("PID gains must be finite and non-negative.")
        if not np.isfinite(self.integral_limit_m_s) or self.integral_limit_m_s <= 0:
            raise ValueError("Integral limit must be finite and positive.")

    def reset(self):
        self.integral_error = 0.0

    def act(self, displacement_m, velocity_m_s, dt_s):
        if dt_s <= 0:
            raise ValueError("dt_s must be positive.")

        x = float(displacement_m)
        v = float(velocity_m_s)

        candidate_integral = float(np.clip(
            self.integral_error + x * dt_s,
            -self.integral_limit_m_s,
            self.integral_limit_m_s,
        ))

        unsaturated_force_n = (
            -self.kp * x
            -self.kd * v
            -self.ki * candidate_integral
        )

        limit_n = self.plant_config.max_control_force_n
        force_n = float(np.clip(unsaturated_force_n, -limit_n, limit_n))

        # Conditional integration: don't integrate further into saturation.
        saturated_high = unsaturated_force_n > limit_n
        saturated_low = unsaturated_force_n < -limit_n
        if not (
            (saturated_high and x < 0)
            or (saturated_low and x > 0)
        ):
            self.integral_error = candidate_integral

        return force_n

"""Baseline controllers for the single-mode vibration-control model."""

from dataclasses import dataclass

import numpy as np

from config import PLANT, PlantConfig


class NoControl:
    """Baseline controller that applies no active force."""

    def reset(self):
        pass

    def act(self, displacement_m, velocity_m_s, dt_s):
        return 0.0


@dataclass
class ActiveDampingController:
    """Velocity-feedback damping in the force-command domain.

    Control law:
        F_control = -velocity_gain * velocity

    The gain is a simulation parameter. It is NOT the actuator-current
    gain reported in the research paper.
    """

    velocity_gain: float = 20.0
    plant_config: PlantConfig = PLANT

    def __post_init__(self):
        if self.velocity_gain < 0:
            raise ValueError("velocity_gain must be non-negative.")

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
    """PID baseline with force saturation and integral anti-windup.

    These are initial simulation gains, not experimentally tuned gains.
    """

    kp: float = 50000.0
    ki: float = 0.0
    kd: float = 20.0
    plant_config: PlantConfig = PLANT
    integral_limit_m_s: float = 1e-3

    integral_error: float = 0.0

    def __post_init__(self):
        if self.kp < 0 or self.ki < 0 or self.kd < 0:
            raise ValueError("PID gains must be non-negative.")

        if self.integral_limit_m_s <= 0:
            raise ValueError("Integral limit must be positive.")

    def reset(self):
        self.integral_error = 0.0

    def act(self, displacement_m, velocity_m_s, dt_s):
        if dt_s <= 0:
            raise ValueError("dt_s must be positive.")

        x = float(displacement_m)
        v = float(velocity_m_s)

        candidate_integral = float(
            np.clip(
                self.integral_error + x * dt_s,
                -self.integral_limit_m_s,
                self.integral_limit_m_s,
            )
        )

        unsaturated_force_n = (
            -self.kp * x
            -self.kd * v
            -self.ki * candidate_integral
        )

        limit_n = self.plant_config.max_control_force_n
        force_n = float(
            np.clip(unsaturated_force_n, -limit_n, limit_n)
        )

        # Conditional integration prevents further windup when saturation
        # would be made worse by integrating the current displacement.
        saturated_high = unsaturated_force_n > limit_n
        saturated_low = unsaturated_force_n < -limit_n

        if not (
            (saturated_high and x < 0)
            or (saturated_low and x > 0)
        ):
            self.integral_error = candidate_integral

        return force_n
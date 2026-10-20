
"""Hybrid stochastic and regenerative cutting-force model."""

import math
import numpy as np

from config import MachiningConfig


class MachiningCuttingForce:
    """Mean, harmonic, stochastic, transient and regenerative cutting force."""

    def __init__(self, config, dt_s, rng=None):
        self.config = config
        self.dt_s = float(dt_s)
        self.rng = rng if rng is not None else np.random.default_rng()
        self._validate()
        self.reset()

    def _validate(self):
        c = self.config

        if self.dt_s <= 0:
            raise ValueError("dt_s must be positive.")
        if c.spindle_rpm <= 0:
            raise ValueError("Spindle RPM must be positive.")
        if c.feed_mm_rev <= 0 or c.depth_of_cut_mm <= 0:
            raise ValueError("Feed and depth of cut must be positive.")
        if c.specific_cutting_force_n_mm2 <= 0:
            raise ValueError("Specific cutting-force coefficient must be positive.")
        if not 0 <= c.force_fluctuation_fraction <= 2:
            raise ValueError("Force fluctuation fraction must be in [0, 2].")
        if c.random_force_fraction < 0:
            raise ValueError("Random force fraction must be non-negative.")
        if not 0 < c.modal_participation <= 1:
            raise ValueError("Modal participation must be in (0, 1].")
        if c.transient_rate_hz < 0 or c.transient_fraction < 0:
            raise ValueError("Transient parameters must be non-negative.")
        if c.transient_decay_s <= 0:
            raise ValueError("Transient decay must be positive.")
        if c.regenerative_feedback_sign not in (-1.0, 1.0):
            raise ValueError("Regenerative feedback sign must be -1 or +1.")

    @property
    def rotational_frequency_hz(self):
        return self.config.spindle_rpm / 60.0

    @property
    def revolution_period_s(self):
        return 1.0 / self.rotational_frequency_hz

    @property
    def delay_samples(self):
        return self.revolution_period_s / self.dt_s

    @property
    def mean_tangential_force_n(self):
        area_mm2 = self.config.feed_mm_rev * self.config.depth_of_cut_mm
        return self.config.specific_cutting_force_n_mm2 * area_mm2

    @property
    def mean_generalized_force_n(self):
        return self.mean_tangential_force_n * self.config.modal_participation

    @property
    def regenerative_gain_n_m(self):
        return (
            self.config.specific_cutting_force_n_mm2
            * self.config.depth_of_cut_mm
            * 1000.0
            * self.config.modal_participation
        )

    def reset(self):
        """Reset delayed motion and stochastic states for a new episode."""
        self.displacement_history = []
        self.sample_index = 0
        self.colored_noise_n = 0.0
        self.transient_amplitude_n = 0.0
        self.transient_age_s = 0.0

        self.harmonic_orders = (1, 2, 3)
        self.harmonic_weights = (1.0, 0.35, 0.18)
        self.harmonic_phases = self.rng.uniform(
            -math.pi, math.pi, size=3
        )

        self.noise_correlation_s = 0.001
        self.noise_rho = math.exp(
            -self.dt_s / self.noise_correlation_s
        )

    def _delayed_displacement(self, current_displacement_m):
        """Interpolate displacement one spindle revolution in the past."""
        n = self.sample_index
        self.displacement_history.append(float(current_displacement_m))

        target_index = n - self.delay_samples

        # During startup, the unknown prehistory is assumed to be zero.
        if target_index < 0:
            return 0.0

        lower = int(math.floor(target_index))
        upper = lower + 1
        fraction = target_index - lower
        history = self.displacement_history

        if lower >= len(history):
            return 0.0
        if upper >= len(history):
            return float(history[lower])

        return float(
            history[lower] * (1.0 - fraction)
            + history[upper] * fraction
        )

    def __call__(self, time_s, displacement_m=0.0):
        c = self.config
        t = float(time_s)
        x_now = float(displacement_m)

        x_delayed = self._delayed_displacement(x_now)

        # 1. Mean generalized cutting load.
        mean_force = self.mean_generalized_force_n

        # 2. Multiple spindle-synchronous harmonics.
        periodic_force = 0.0
        harmonic_scale = mean_force * c.force_fluctuation_fraction
        base_frequency = self.rotational_frequency_hz

        for order, weight, phase in zip(
            self.harmonic_orders,
            self.harmonic_weights,
            self.harmonic_phases,
        ):
            periodic_force += (
                harmonic_scale
                * weight
                * math.sin(
                    2.0 * math.pi * order * base_frequency * t
                    + float(phase)
                    + c.phase_rad
                )
            )

        # 3. Temporally correlated noise using an AR(1) process.
        noise_std = mean_force * c.random_force_fraction
        innovation_std = noise_std * math.sqrt(
            max(0.0, 1.0 - self.noise_rho**2)
        )
        self.colored_noise_n = (
            self.noise_rho * self.colored_noise_n
            + innovation_std * self.rng.normal()
        )

        # 4. Random transient bursts with exponential decay.
        if self.rng.random() < c.transient_rate_hz * self.dt_s:
            sign = -1.0 if self.rng.random() < 0.5 else 1.0
            self.transient_amplitude_n = (
                sign
                * mean_force
                * c.transient_fraction
                * self.rng.uniform(0.5, 1.5)
            )
            self.transient_age_s = 0.0

        transient_force = (
            self.transient_amplitude_n
            * math.exp(
                -self.transient_age_s / c.transient_decay_s
            )
        )
        self.transient_age_s += self.dt_s

        # 5. Regenerative force from current versus previous-revolution motion.
        regenerative_force = (
            c.regenerative_feedback_sign
            * self.regenerative_gain_n_m
            * (x_now - x_delayed)
        )

        total_force = (
            mean_force
            + periodic_force
            + self.colored_noise_n
            + transient_force
            + regenerative_force
        )

        self.sample_index += 1
        return float(total_force)

    def describe(self):
        c = self.config
        return {
            "spindle_rpm": float(c.spindle_rpm),
            "rotational_frequency_hz": float(self.rotational_frequency_hz),
            "revolution_period_s": float(self.revolution_period_s),
            "delay_samples": float(self.delay_samples),
            "feed_mm_rev": float(c.feed_mm_rev),
            "depth_of_cut_mm": float(c.depth_of_cut_mm),
            "specific_cutting_force_n_mm2": float(
                c.specific_cutting_force_n_mm2
            ),
            "mean_tangential_force_n": float(self.mean_tangential_force_n),
            "mean_generalized_force_n": float(self.mean_generalized_force_n),
            "regenerative_gain_n_m": float(self.regenerative_gain_n_m),
            "force_fluctuation_fraction": float(
                c.force_fluctuation_fraction
            ),
            "random_force_fraction": float(c.random_force_fraction),
            "modal_participation": float(c.modal_participation),
            "transient_rate_hz": float(c.transient_rate_hz),
            "regenerative_feedback_sign": float(
                c.regenerative_feedback_sign
            ),
        }

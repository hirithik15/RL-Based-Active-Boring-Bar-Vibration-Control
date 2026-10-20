
"""Configuration for the boring-bar vibration-control simulation.

The plant is a reduced-order single-mode model. The cutting-force model
includes stochastic variation and regenerative displacement feedback,
but requires calibration before being treated as a physical digital twin.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PlantConfig:
    natural_frequency_hz: float = 188.0
    damping_ratio: float = 0.034
    modal_stiffness_n_m: float = 2.6e6
    dt_s: float = 1.0e-4
    max_control_force_n: float = 2.0

    @property
    def natural_frequency_rad_s(self) -> float:
        return 2.0 * math.pi * self.natural_frequency_hz

    @property
    def modal_mass_kg(self) -> float:
        return self.modal_stiffness_n_m / self.natural_frequency_rad_s**2

    @property
    def stiffness_n_m(self) -> float:
        return self.modal_stiffness_n_m

    @property
    def damping_ns_m(self) -> float:
        return (
            2.0
            * self.damping_ratio
            * self.modal_mass_kg
            * self.natural_frequency_rad_s
        )


@dataclass(frozen=True)
class DisturbanceConfig:
    """Legacy harmonic disturbance configuration."""
    amplitude_n: float = 1.0
    frequency_hz: float = 188.0
    phase_rad: float = 0.0


@dataclass(frozen=True)
class MachiningConfig:
    """Machining parameters and force-model uncertainty for one episode.

    specific_cutting_force_n_mm2:
        Effective specific cutting-force coefficient.

    modal_participation:
        Fraction mapping cutting force into the modeled modal coordinate.

    regenerative_feedback_sign:
        +1 selects the destabilizing current-minus-delayed displacement
        convention; -1 reverses that convention. The physical sign must
        ultimately be established from the chosen coordinate system.
    """
    spindle_rpm: float = 9000.0
    feed_mm_rev: float = 0.10
    depth_of_cut_mm: float = 0.50
    specific_cutting_force_n_mm2: float = 1800.0

    force_fluctuation_fraction: float = 0.20
    random_force_fraction: float = 0.08
    modal_participation: float = 0.02

    transient_rate_hz: float = 3.0
    transient_fraction: float = 0.50
    transient_decay_s: float = 0.003

    regenerative_feedback_sign: float = 1.0
    phase_rad: float = 0.0


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    total_timesteps: int = 300_000
    episode_duration_s: float = 0.2

    learning_rate: float = 3e-4
    buffer_size: int = 200_000
    learning_starts: int = 5_000
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1

    # Legacy disturbance ranges retained for compatibility.
    force_amplitude_min_n: float = 0.5
    force_amplitude_max_n: float = 1.5
    force_frequency_min_hz: float = 150.0
    force_frequency_max_hz: float = 200.0

    # Randomized machining ranges.
    spindle_rpm_min: float = 600.0
    spindle_rpm_max: float = 12000.0
    feed_min_mm_rev: float = 0.05
    feed_max_mm_rev: float = 0.20
    depth_min_mm: float = 0.20
    depth_max_mm: float = 1.00

    specific_cutting_force_min_n_mm2: float = 1400.0
    specific_cutting_force_max_n_mm2: float = 2200.0

    force_fluctuation_min: float = 0.10
    force_fluctuation_max: float = 0.35
    random_force_fraction_min: float = 0.03
    random_force_fraction_max: float = 0.12
    modal_participation_min: float = 0.01
    modal_participation_max: float = 0.04

    transient_rate_min_hz: float = 0.0
    transient_rate_max_hz: float = 8.0
    transient_fraction_min: float = 0.20
    transient_fraction_max: float = 0.80

    # Initial-state randomization.
    initial_displacement_max_m: float = 1e-5
    initial_velocity_max_m_s: float = 0.005

    # Observation normalization scales.
    displacement_scale_m: float = 1e-5
    velocity_scale_m_s: float = 0.05

    displacement_reward_weight: float = 1.0
    control_effort_reward_weight: float = 0.01
    action_change_reward_weight: float = 0.005


@dataclass(frozen=True)
class EvaluationConfig:
    duration_s: float = 0.2

    # Nominal and shifted operating conditions.
    nominal_spindle_rpm: float = 9000.0
    unseen_spindle_rpm: float = 11280.0
    feed_mm_rev: float = 0.10
    depth_of_cut_mm: float = 0.50

    specific_cutting_force_n_mm2: float = 1800.0
    force_fluctuation_fraction: float = 0.20
    random_force_fraction: float = 0.08
    modal_participation: float = 0.02
    transient_rate_hz: float = 3.0
    transient_fraction: float = 0.50

    # Retained for compatibility with older scripts.
    nominal_amplitude_n: float = 1.0
    nominal_frequency_hz: float = 188.0
    unseen_amplitude_n: float = 1.4
    unseen_frequency_hz: float = 205.0
    test_phase_rad: float = 0.73
    plot_time_window_s: float = 0.2


PLANT = PlantConfig()
DISTURBANCE = DisturbanceConfig()
TRAINING = TrainingConfig()
EVALUATION = EvaluationConfig()

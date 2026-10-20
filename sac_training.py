
"""Train and save a Soft Actor-Critic vibration controller."""
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from config import PLANT, TRAINING
from env import VibrationEnv


def train_sac(
    total_timesteps=None,
    seed=None,
    model_path="results/sac_machining_aware_v1",
):
    """Train SAC and save the trained model.

    Parameters can be overridden for short smoke tests.
    """
    if total_timesteps is None:
        total_timesteps = TRAINING.total_timesteps
    if seed is None:
        seed = TRAINING.seed

    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive.")

    set_random_seed(seed)

    env = Monitor(
        VibrationEnv(
            plant_config=PLANT,
            training_config=TRAINING,
            randomize_episode=True,
        )
    )

    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=TRAINING.learning_rate,
        buffer_size=TRAINING.buffer_size,
        learning_starts=TRAINING.learning_starts,
        batch_size=TRAINING.batch_size,
        tau=TRAINING.tau,
        gamma=TRAINING.gamma,
        train_freq=TRAINING.train_freq,
        gradient_steps=TRAINING.gradient_steps,
        seed=seed,
        verbose=1,
        device="auto",
        tensorboard_log="results/tensorboard",
    )

    print(f"Starting SAC training for {total_timesteps:,} timesteps.")
    model.learn(
        total_timesteps=int(total_timesteps),
        progress_bar=False,
    )

    output_path = Path(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))

    env.close()
    print(f"Saved SAC model to: {output_path}.zip")
    return model


if __name__ == "__main__":
    train_sac(
        total_timesteps=300_000,
        seed=42,
        model_path="results/sac_machining_aware_v1",
    )

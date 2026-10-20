import matplotlib.pyplot as plt
import numpy as np
from env import VibrationEnv

env = VibrationEnv(randomize_episode=True)
obs, info = env.reset(seed=42)

times = []
forces = []

for _ in range(env.max_steps):
    obs, reward, terminated, truncated, step_info = env.step([0.0])
    times.append(step_info["time_s"] * 1000)
    forces.append(step_info["disturbance_force_n"])

    if terminated or truncated:
        break

plt.figure(figsize=(10, 4))
plt.plot(times, forces)
plt.xlabel("Time (ms)")
plt.ylabel("Generalized cutting force (N)")
plt.title("Stochastic cutting force with regenerative feedback")
plt.grid(True)
plt.tight_layout()
plt.show()

env.close()
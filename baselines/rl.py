import torch
from stable_baselines3 import PPO, SAC

import gym_env  # If this fails, install it with `pip install -e .` from \simulator\gym_env
import gymnasium as gym
import numpy as np

if __name__ == "__main__":
    env = gym.make("SimulatorEnv-v0")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # model = SAC("MlpPolicy", env, verbose=1, device=device)
    # model.learn(total_timesteps=200_000, log_interval=4, progress_bar=True)
    # model.save("sac_5_rl")
    # del model  # remove to demonstrate saving and loading
    model = SAC.load("sac_5_rl")

    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            env.render()
            obs, info = env.reset()

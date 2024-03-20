from stable_baselines3 import PPO

import gym_env  # If this fails, install it with `pip install -e .` from \simulator\gym_env
import gymnasium as gym
import numpy as np

if __name__ == "__main__":
    env = gym.make("SimulatorEnv-v0")
    episode_rewards = []
    episode_lengths = []

    # model = PPO("MlpPolicy", env, verbose=1)
    # model.learn(total_timesteps=1_000_000, log_interval=4, progress_bar=True)
    # model.save("ppo")
    # del model  # remove to demonstrate saving and loading  # TODO
    model = PPO.load("ppo")

    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            env.render()
            obs, info = env.reset()
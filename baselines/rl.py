import os

import gymnasium as gym


import numpy as np
import torch
from baselines.sac.model import Actor as SACActor

import wandb
from wandb.integration.sb3 import WandbCallback

import gym_env  # If this fails, install it with `pip install -e .` from \simulator\gym_env


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Generate random seeds
    num_seeds = 3
    seeds = np.random.randint(0, 10000, size=num_seeds).tolist()

    # Configuration dictionary
    config = {
        "env": "SimulatorEnv-v0",
        "total_timesteps": 1_000_000,
        "log_interval": 4,
        "progress_bar": True,
        "device": device,
        "policy_type": "MlpPolicy",
        "seeds": seeds  # Use generated seeds
    }

    def evaluate_model(model, env, num_episodes=10):
        total_rewards = []
        for _ in range(num_episodes):
            # TODO: use fixed seeding for evaluation episodes
            obs = env.reset()
            done = False
            total_reward = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                total_reward += reward
            total_rewards.append(total_reward)
        return sum(total_rewards) / num_episodes


    env = gym.make(config['env'], dataset_split="test")

    model = SACActor(env, device=device).to(device)
    model.load_state_dict(torch.load("best_model_sac_SimulatorEnv-v0__model__1__1721400747.pth"))
    model.eval()

    obs, info = env.reset()
    while True:
        action = model.act(torch.Tensor(obs).to(device), deterministic=True)
        action = action.detach().cpu().numpy()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            env.render()
            obs, info = env.reset()

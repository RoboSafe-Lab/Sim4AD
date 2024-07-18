import os

import gymnasium as gym


import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

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

    # Initialize WandB project
    run = wandb.init(project="sim4ad", config=config, sync_tensorboard=True, monitor_gym=True, save_code=True)

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


    best_model = None
    best_reward = -float('inf')

    for seed in config['seeds']:
        try:
            env = Monitor(gym.make(config['env']), filename=None)
            env.seed(seed)
            np.random.seed(seed)

            model = SAC(config['policy_type'], env, verbose=1, device=config['device'],
                        tensorboard_log=f"runs/{run.id}/seed_{seed}")

            model.learn(total_timesteps=config['total_timesteps'], log_interval=config['log_interval'],
                        progress_bar=config['progress_bar'],
                        callback=WandbCallback(gradient_save_freq=100, model_save_path=f"models/{run.id}/seed_{seed}",
                                               verbose=2))

            # Evaluate the model
            avg_reward = evaluate_model(model, env)

            # Save the model if it's the best one
            if avg_reward > best_reward:
                best_reward = avg_reward
                best_model = model
                best_model_path = f"models/{run.id}/best_model"
        except Exception as e:
            print(f"Error in seed {seed}: {e}")
            continue

    # Save the best model
    if best_model:
        best_model.save(best_model_path)
        print(f"Best model saved with average reward: {best_reward}")

    wandb.finish()

    # # del model  # remove to demonstrate saving and loading
    # model = SAC.load("sac_5_rl")
    #
    # obs, info = env.reset()
    # while True:
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     if terminated or truncated:
    #         env.render()
    #         obs, info = env.reset()

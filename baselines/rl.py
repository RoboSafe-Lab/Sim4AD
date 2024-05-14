import os

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

import gym_env  # If this fails, install it with `pip install -e .` from \simulator\gym_env
import gymnasium as gym


if __name__ == "__main__":


    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Separate evaluation env
    env = Monitor(gym.make("SimulatorEnv-v0"), filename=None)
    eval_env = Monitor(gym.make("SimulatorEnv-v0"), filename=None)
    # Use deterministic actions for evaluation
    eval_callback = EvalCallback(eval_env, best_model_save_path='baselines/sac/',
                                 log_path='baselines/sac/logs/', eval_freq=500,
                                 deterministic=True, render=False)

    model = SAC("MlpPolicy", env, verbose=1, device=device, tensorboard_log="baselines/runs/sac_tensorboard/")
    model.learn(total_timesteps=1_000_000, log_interval=4, progress_bar=True, callback=eval_callback)
    model.save("maybe_not_best_sac_5_rl") ## TODO: check if baseline/sac has a better one!
    # del model  # remove to demonstrate saving and loading
    # model = SAC.load("sac_5_rl")
    #
    # obs, info = env.reset()
    # while True:
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     if terminated or truncated:
    #         env.render()
    #         obs, info = env.reset()

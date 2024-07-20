from sim4ad.path_utils import get_file_name_trajectories, get_path_to_automatum_scenario, get_path_to_automatum_map, \
    get_config_path, get_path_offlinerl_model
from simulator.lightweight_simulator import Sim4ADSimulation
from sim4ad.offlinerlenv.td3bc_automatum import TD3_BC_TrainerLoader, TrainConfig, normalize_states

import os
import pickle
from typing import List
import torch

import numpy as np
from tqdm import tqdm

import gymnasium as gym
import gym_env  # If this fails, install it with `pip install -e .` from \simulator\gym_env

"""
This file si the one that uses the simulator in one of the three analyses types we have, generates the corresponding
trajectories xi that try to replicate demonstrations D, and then evaluates the performance of the policy pi
"""


def generate_trajectories(policy_type, spawn_method, irl_weights, episode_name:List[str]):
    """
    Generate trajectories using the simulator and the given policy.

    :param simulator: The simulator to use.
    :param policy_type: The policy to use.
    :param irl_weights: The IRL weights to use.
    :param output_dir: The output directory.
    :param num_trajectories: The number of trajectories to generate.
    """

    if "cluster" in policy_type:
        cluster = policy_type.split("_")[-1]
    else:
        cluster = "all"

    # TODO sim = Sim4ADSimulation(episode_name=episode_name, policy_type=policy_type, spawn_method=spawn_method, clustering=cluster)

    config = {
        "env": "SimulatorEnv-v0",
        "env_kwargs": {
            'clustering': cluster
        }
    }

    gym_env = gym.make(config['env'], **config['env_kwargs'])
    model_path = get_path_offlinerl_model()
    trainer_loader = TD3_BC_TrainerLoader(TrainConfig)
    trainer_loader.load_model(model_path)
    trainer_loader.actor.eval()

    simulation_length = "done" # seconds

    output_dir = get_file_name_trajectories(policy_type, spawn_method, irl_weights, episode_name, simulation_length)
    if os.path.exists(output_dir):
        print(f"Trajectories already exist for {episode_name} in mode {spawn_method} w/ policy {policy_type}."
              f"\n==> LOADING!!!! ")
        with open(output_dir, "rb") as f:
            return pickle.load(f)


    # for _ in tqdm(range(int(np.floor(simulation_length / sim.dt))), desc=f"Running {episode_name} in mode {spawn_method}"
    #                                                                     f" w/ policy {policy_type}"):
    #     sim.step()
    looped_dataset = False # Whether we have collected al the possible trajectories/episodes
    steps = 0

    assert simulation_length == "done"
    while not looped_dataset:
        obs, info = gym_env.reset(seed=0) # TODO: set seed!

        episode_done = False
        while not episode_done:
            # TODO done = sim.step(return_done=True)
            action = trainer_loader.actor(torch.tensor(obs, dtype=torch.float32, device='mps')) # TODO: deterministic=True) # TODO: deterministic?
            action = action.cpu().detach().numpy()
            next_obs, reward, terminated, truncated, info = gym_env.step(action)
            done = terminated or truncated
            obs = next_obs

            steps += 1

            if steps % 1000 == 0:
                print(f"Simulation time {gym_env.unwrapped.simulation.time} in {episode_name} in mode {spawn_method} w/ policy {policy_type}")

            if done:
                break

        looped_dataset = gym_env.unwrapped.simulation.done_full_cycle
        gym_env.unwrapped.simulation.replay_simulation()

    gym_env.unwrapped.simulation.kill_all_agents()

    simulation_agents =gym_env.unwrapped.simulation.evaluator.get_picklable_agents()

    output_dir = get_file_name_trajectories(policy_type, spawn_method, irl_weights, episode_name, simulation_length)
    with open(output_dir, "wb") as f:
        pickle.dump(simulation_agents, f)
    return simulation_agents


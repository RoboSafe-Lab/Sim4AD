from sim4ad.path_utils import get_file_name_trajectories
from simulator.lightweight_simulator import Sim4ADSimulation

import os
import pickle
from typing import List

import numpy as np
from tqdm import tqdm


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

    sim = Sim4ADSimulation(episode_name=episode_name, policy_type=policy_type, spawn_method=spawn_method, clustering=cluster)

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
    done = False
    steps = 0
    assert simulation_length == "done"
    while not done:
        done = sim.step(return_done=True)
        steps += 1

        if steps % 1000 == 0:
            print(f"Simulation time {sim.time} in {episode_name} in mode {spawn_method} w/ policy {policy_type}")

        if done:
            break

    sim.kill_all_agents()

    simulation_agents = sim.evaluator.get_picklable_agents()

    output_dir = get_file_name_trajectories(policy_type, spawn_method, irl_weights, episode_name, simulation_length)
    with open(output_dir, "wb") as f:
        pickle.dump(simulation_agents, f)
    return simulation_agents

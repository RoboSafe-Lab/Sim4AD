import gymnasium as gym
import numpy as np
from tqdm import tqdm
import torch
import os
import pickle
from loguru import logger

from sim4ad.path_utils import get_config_path, get_file_name_trajectories, get_path_offlinerl_model, get_path_sac_model
from sim4ad.data import ScenarioConfig
from simulator.lightweight_simulator import Sim4ADSimulation
from sim4ad.offlinerlenv.td3bc_automatum import TD3_BC_Loader
from baselines.sac.model import Actor as SACActor
from evaluation.evaluation_functions import EvaluationFeaturesExtractor


class HumanLikenessEvaluation:
    """Evaluate offline RL and SAC for human likeness analysis"""

    def __init__(self, eval_configs, evaluation_episodes, policy_type, cluster, driving_style_model_paths, load_policy):

        dummy_env = gym.make(eval_configs.env_name, dummy=True)

        driving_styles = {}
        for style in driving_style_model_paths:
            model_path = driving_style_model_paths[style]
            if model_path is not None:
                driving_styles[style] = load_policy(policy_type, style, dummy_env, device=eval_configs.device,
                                                    eval_configs=eval_configs, evaluation_episodes=evaluation_episodes)
                driving_styles[style].eval()

        self.sim = Sim4ADSimulation(episode_name=evaluation_episodes, spawn_method=eval_configs.spawn_method,
                                    policy_type=policy_type.name, clustering=cluster,
                                    driving_style_policies=driving_styles)

    def simulation(self, visualization: bool = True):
        """Using the simulator to simulate trajectories of all agents"""
        self.sim.full_reset()
        # done = False # TODO: uncomment this to run until we use all vehicles
        # while not done:
        #     assert spawn_method != "random", "we will never finish!"
        #     done = sim.step(return_done=True)

        simulation_length = 50  # seconds
        for _ in tqdm(range(int(np.floor(simulation_length / self.sim.dt)))):
            self.sim.step()

        # remove all agents left in the simulation.
        self.sim.kill_all_agents()
        if visualization:
            self.sim.replay_simulation()

        simulation_agents = self.sim.evaluator.get_picklable_agents()

        return simulation_agents


if __name__ == "__main__":
    raise NotImplementedError("This script is not meant to be run directly. Please run `run_evaluations.py` instead.")

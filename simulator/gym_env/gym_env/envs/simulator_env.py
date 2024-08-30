"""
Gymnasium based environment that is based on the lightweight simulator.
Based on the official tutorial: https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/
"""
import os
import pickle
from typing import List

import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
from openautomatumdronedata.dataset import droneDataset

from sim4ad.common_constants import DEFAULT_CLUSTER
from sim4ad.util import load_dataset
from sim4ad.data import DatasetScenario, ScenarioConfig
from loguru import logger
from sim4ad.opendrive import Map
from sim4ad.path_utils import get_path_to_automatum_scenario, get_config_path, get_path_to_automatum_map, \
    get_path_irl_weights, get_common_property
from simulator.lightweight_simulator import Sim4ADSimulation

from simulator.gym_env.gym_env.envs.reward import get_reward


class SimulatorEnv(gym.Env):
    """
    Gym environment for the lightweight simulator.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode: str = None, episode_names: List = None, seed: int = None, clustering: str = "All",
                 dataset_split: str = "train", use_irl_reward: bool = False, spawn_method: str = None, dummy: bool = False):
        """
        Args:
            config: Configuration for the environment.
            dummy: If True, the environment will not load the dataset and will only be used for getting the dimensions
        """

        # Load "common elements" to get the action and observation space
        obs_features = list(get_common_property("FEATURES_IN_OBSERVATIONS"))
        action_features = list(get_common_property("FEATURES_IN_ACTIONS"))

        # At each step, the agent must choose the acceleration and yaw rate.
        self.MIN_YAW_RATE = -0.08
        self.MAX_YAW_RATE = 0.08

        assert action_features == ["acceleration", "yaw_rate"], (f"Action features are not as expected: "
                                                                  f"{action_features}. Change below.")
        self.action_space = Box(
            low=np.array([-5, self.MIN_YAW_RATE]),  # acceleration, yaw_rate
            high=np.array([5, self.MAX_YAW_RATE])
        )

        obs_space = len(obs_features)
        self.observation_space = Box(
            low=np.array([-np.inf] * obs_space),  # velocity can max be 200, heading can min be -pi
            high=np.array([np.inf] * obs_space)
        )

        if dummy:
            # We only need the action and observation space
            return

        assert spawn_method is not None, "Please provide the spawn method"
        self.spawn_method = spawn_method

        assert clustering in ["All", "Aggressive", "Normal", "Cautious"], \
            "Invalid clustering. If you used 'General', please use 'All' instead."

        self.weights = None
        if episode_names is None:
            logger.warning("No episode names provided. Loading the DEFAULT dataset. MAP is: appershofen")
            # Load all episodes in the training dataset
            configs = ScenarioConfig.load(get_config_path("appershofen"))  # todo: change scenario name
            idx = configs.dataset_split[dataset_split]
            self.episode_names = [x.recording_id for i, x in enumerate(configs.episodes) if i in idx]
        else:
            self.episode_names = episode_names

        self.simulation = Sim4ADSimulation(episode_name=self.episode_names, policy_type="rl",
                                           simulation_name=f"gym_sim_{self.episode_names}",
                                           spawn_method=self.spawn_method, clustering=clustering)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self._dataset_split = dataset_split

        if use_irl_reward:
            self.load_weights()
        self.seed_used = seed

    @property
    def driving_style(self):
        return self.simulation.clustering

    @property
    def dataset_split(self):
        return self._dataset_split

    @property
    def map_name(self):
        return self.simulation.map_name

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.simulation.seed(seed)
        self.seed_used = seed
        return [seed]

    def load_weights(self):
        # Load the weights from IRL
        cluster = self.simulation.clustering
        with open(get_path_irl_weights(cluster), "rb") as f:
            self.weights = pickle.load(f)['theta'][-1]

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random

        if seed is None:
            seed = self.seed_used

        if seed is not None:
            super().reset(seed=seed)
            self.seed(seed)

        obs, info = self.simulation.soft_reset()
        return np.array(obs, dtype=np.float32), info

    def step(self, action):
        """

        :param action: (acceleration, yaw_rate)
        :return:
        """

        # Info contains the cause of the death, if dead
        next_obs, info = self.simulation.step(action)
        terminated = info["reached_goal"]
        truncated = info["truncated"] or info["collision"] or info["off_road"]

        # An episode is terminated iff the agent has reached the target
        reward = get_reward(terminated=terminated, truncated=truncated, info=info, irl_weights=self.weights)

        return np.array(next_obs, dtype=np.float32), reward, terminated, truncated, info

    def render(self):
        """
        Replay an EPISODE.
        """
        self.simulation.replay_simulation()

    def close(self):

        """
        In other environments close might also close files that were opened or release other resources.
        You shouldn’t interact with the environment after having called close.

        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
        """
        return NotImplementedError






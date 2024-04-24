"""
Gymnasium based environment that is based on the lightweight simulator.
Based on the official tutorial: https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/
"""
import os
import pickle

import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
from openautomatumdronedata.dataset import droneDataset

from sim4ad.util import load_dataset
from sim4ad.data import DatasetScenario, ScenarioConfig
from sim4ad.opendrive import Map
from sim4ad.path_utils import get_path_to_automatum_scenario, get_config_path, get_path_to_automatum_map, \
    get_path_irl_weights
from simulator.lightweight_simulator import Sim4ADSimulation

from simulator.gym_env.gym_env.envs.reward import get_reward


class SimulatorEnv(gym.Env):
    """
    Gym environment for the lightweight simulator.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}
    SPAWN_METHOD = "dataset_one"  # We assume there is a

    def __init__(self, render_mode: str = None, config: dict = None):
        """
        Args:s
            config: Configuration for the environment.
        """

        if config is None:
            # Load all episodes in the training dataset
            configs = ScenarioConfig.load(get_config_path("appershofen"))
            idx = configs.dataset_split["train"]
            self.episode_names = [x.recording_id for i, x in enumerate(configs.episodes) if i in idx]
        else:
            raise NotImplementedError

        self.simulation = Sim4ADSimulation(episode_name=self.episode_names, policy_type="rl",
                                           simulation_name=f"gym_sim_{self.episode_names}",
                                           spawn_method=self.SPAWN_METHOD)

        # At each step, the agent must choose the acceleration and steering angle.
        self.action_space = Box(
            low=np.array([-5, -np.pi]),
            high=np.array([5, np.pi])
        )

        # The observation will be a set of 34 features.
        self.observation_space = Box(
            low=np.array([-np.nan] * 34),
            high=np.array([np.nan] * 34)
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Load the weights from IRL from training_log.pkl
        with open(get_path_irl_weights(), "rb") as f:
            self.weights = pickle.load(f)

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        obs, info = self.simulation.soft_reset()
        return obs, info

    def step(self, action):
        """

        :param action: (acceleration, steering_angle)
        :return:
        """

        # Info contains the cause of the death, if dead
        next_obs, info = self.simulation.step(action)
        terminated = info["reached_goal"]
        truncated = info["truncated"] or info["collision"] or info["off_road"]

        # An episode is terminated iff the agent has reached the target
        # technically reward is a function of (s, a, s')...
        reward = get_reward(terminated=terminated, truncated=truncated, info=info, irl_weights=self.weights)

        return next_obs, reward, terminated, truncated, info

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






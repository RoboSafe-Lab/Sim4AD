import os
import pickle
from typing import List, Dict
import numpy as np
from loguru import logger

from gymnasium.spaces import Box
from pettingzoo.utils import ParallelEnv

from sim4ad.data import ScenarioConfig
from sim4ad.path_utils import get_config_path, get_path_irl_weights, get_common_property
from simulator.lightweight_simulator import Sim4ADSimulation
from simulator.gym_env.gym_env.envs.reward import get_reward

class MultiCarEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, 
                 render_mode: str = None,
                 episode_names: List = None,
                 seed: int = None,
                 clustering: str = "All",
                 dataset_split: str = "train",
                 use_irl_reward: bool = False,
                 spawn_method: str = None,
                 evaluation: bool = True,
                 max_steps: int = 1000,
                 dummy: bool = False):
        """
        Parallel multi-agent environment that uses Sim4ADSimulation for multiple agents.
        It expects that Sim4ADSimulation is modified to support multi-agent RL:
        - `soft_reset_multi()` to initiate an episode and return all initial agents and observations
        - `step_multi(actions_dict)` to advance by one step given a dict of actions for each agent
        """

        # Load "common elements" to get the action and observation space
        obs_features = list(get_common_property("FEATURES_IN_OBSERVATIONS"))
        action_features = list(get_common_property("FEATURES_IN_ACTIONS"))
        assert action_features == ["acceleration", "yaw_rate"], "Expected action features ['acceleration', 'yaw_rate']."

        # action space
        self.MIN_YAW_RATE = -0.08
        self.MAX_YAW_RATE = 0.08
        self._action_space = Box(low=np.array([-5, self.MIN_YAW_RATE], dtype=np.float32),
                                 high=np.array([5, self.MAX_YAW_RATE], dtype=np.float32))

        # observation space
        obs_dim = len(obs_features)
        self._observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.use_irl_reward = use_irl_reward
        self.render_mode = render_mode
        self._dataset_split = dataset_split
        self.seed_used = seed
        self.max_steps = max_steps

        if dummy:
            self.simulation = None
            self.agents = []
            return

        assert spawn_method is not None, "Please provide the spawn_method."
        assert clustering in ["All", "Aggressive", "Normal", "Cautious"], "Invalid clustering."

        if not episode_names:
            logger.warning("No episode names provided. Loading default scenario.")
            configs = ScenarioConfig.load(get_config_path("appershofen"))
            idx = configs.dataset_split[dataset_split]
            episode_names = [x.recording_id for i, x in enumerate(configs.episodes) if i in idx]

        self.simulation = Sim4ADSimulation(
            episode_name=episode_names,
            policy_type="rl",  
            simulation_name="multi_gym_sim",
            spawn_method=spawn_method,
            clustering=clustering,
            evaluation=evaluation,
            normalise_observation=True,
            pbar=False
        )

        if self.use_irl_reward:
            cluster = self.simulation.clustering
            with open(get_path_irl_weights(cluster), "rb") as f:
                self.weights = pickle.load(f)['theta'][-1]
        else:
            self.weights = None

        self.agents = []
        self.terminated_agents = set()
        self.truncated_agents = set()
        self.current_step = 0

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def seed(self, seed=None):
        if seed is not None:
            self.seed_used = seed
        if self.seed_used is not None:
            np.random.seed(self.seed_used)
            self.simulation.seed(self.seed_used)

    def reset(self, seed=None, options=None):
        if seed is None:
            seed = self.seed_used

        if seed is not None:
            super().reset(seed=seed)
            self.seed(seed)

        # use simulation的soft_reset_multi()
        obs_dict, _ = self.simulation.soft_reset_multi()

        self.agents = obs_dict.keys()
        self.terminated_agents = set()
        self.truncated_agents = set()
        self.current_step = 0

        return obs_dict

    def step(self, actions: Dict[str, np.ndarray]):
        next_obs_all, terminated_all, truncated_all, info_all = self.simulation.step_multi(actions)
        current_agent_ids = list(next_obs_all.keys())
        obs_dict = {}
        rewards_dict = {}
        dones_dict = {}
        infos_dict = {}

        alive_agents = []
        #agent_ids = self.agents

        # reward
        rewards_all = []
        #trunc = truncated_all
        #term = terminated_all
        for aid in current_agent_ids:
            term = terminated_all.get(aid, False)  
            trunc = truncated_all.get(aid, False)  

            # reward
            reward = get_reward(terminated=term, truncated=trunc, info=info_all[aid], irl_weights=self.weights)
            rewards_all.append(reward)

            if term or trunc:
                dones_dict[aid] = True
                rewards_dict[aid] = reward
                obs_dict[aid] = np.zeros(self.observation_space.shape, dtype=np.float32)
            else:
                obs_dict[aid] = next_obs_all[aid]
                rewards_dict[aid] = reward
                dones_dict[aid] = False
                alive_agents.append(aid)

            infos_dict[aid] = info_all[aid]
        self.current_step += 1
        #done_all = any(truncated_all[i] for i in range(len(agent_ids)))
        done_all = any(truncated_all.get(aid, False) for aid in current_agent_ids)
        dones_dict["__any__"] = done_all

        if done_all:
            # 全部结束
            pass
        else:
            self.agents = alive_agents

        return obs_dict, rewards_dict, dones_dict, infos_dict


    def render(self):
        self.simulation.replay_simulation()

    def close(self):
        pass
    
    def is_done_full_cycle(self):
        """Expose done_full_cycle from the simulation."""
        return self.simulation.done_full_cycle
    
    def current_time(self):
        return self.simulation.simulation_time
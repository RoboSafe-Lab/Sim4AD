"""
Define the reward function for the environment. The reward function is a callable that takes the following arguments:

        state: The current state of the environment.
        action: The action taken by the agent.
        next_state: The next state of the environment.
        done: A boolean indicating whether the episode has ended.
        info: A dictionary containing any additional information.
"""
import pickle

import numpy as np

from sim4ad.path_utils import get_path_irl_weights


def get_reward(terminated, truncated, info, irl_weights=None):
    """
    Reward function for the environment.
    Currently, it is a sparse reward function that is zero but when agent reaches the goal, collides or goes off-road.
    """

    use_rl_reward = False
    if use_rl_reward:

        features = np.array([info["ego_speed"], abs(info["ego_long_acc"]), abs(info["ego_lat_acc"]),
                             abs(info["ego_long_jerk"]), info["thw_front"], info["thw_rear"], info["collision"],
                             info["induced_deceleration"]])

        assert irl_weights is not None, "IRL weights are not provided."
        # Compute the reward
        return np.dot(irl_weights["theta"][0], features)

    else:
        if terminated:
            return 1
        elif truncated and info["collision"]:
            return -1
        elif truncated and info["off_road"]:
            return -1
        elif truncated:
            # The episode was truncated due to the maximum number of steps being reached.
            return -1
        else:
            return 0


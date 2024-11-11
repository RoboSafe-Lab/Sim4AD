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
from sim4ad.common_constants import DEFAULT_DECELERATION_VALUE


def get_reward(terminated, truncated, info, irl_weights=None):
    """
    Reward function for the environment.
    Currently, it is a sparse reward function that is zero but when agent reaches the goal, collides or goes off-road.
    """

    if irl_weights is not None:

        # TODO Check that the alternative values are correct: e.g., that if None the value should be 0 or 1
        speed = np.exp(-1 / abs(info["ego_speed"])) if info["ego_speed"] else 0
        long_acc = np.exp(-1 / abs(info["ego_long_acc"])) if info["ego_long_acc"] else 0
        lat_acc = np.exp(-1 / abs(info["ego_lat_acc"])) if info["ego_lat_acc"] else 0
        long_jerk = np.exp(-1 / abs(info["ego_long_jerk"])) if info["ego_long_jerk"] else 0
        thw_front = np.exp(-abs(info["thw_front"])) if info["thw_front"] else 1
        thw_rear = np.exp(-abs(info["thw_rear"]))   if info["thw_rear"] else 1
        d_centerline = np.exp(-1 / abs(info["d_centerline"])) if info["d_centerline"] else 0
        # d_centerline = np.exp(-1 / abs(info["d_centerline"])) if info["d_centerline"] else 0
        #nearest_distance_lane_marking = np.exp(-1 / abs(info["nearest_distance_lane_marking"])) \
            #if info["nearest_distance_lane_marking"] else 0
        lane_deviation_rate = np.exp(-1 / abs(info["lane_deviation_rate"])) if info["lane_deviation_rate"] else 0
        left_lane_available =  info["left_lane_available"] 
        right_lane_available = info["right_lane_available"] 
        print(f"the second check: {left_lane_available}, the second check: {right_lane_available}")
        features = np.array([speed, long_acc, lat_acc, long_jerk,
                             thw_front, thw_rear, d_centerline, lane_deviation_rate, left_lane_available, right_lane_available])
        for i, value in enumerate(features):
            if value is None:
                print(f"Warning: Feature at index {i} is None.")

        assert all([0 <= f <= 1 for f in features]), "Features should be between 0 and 1."

        assert irl_weights is not None, "IRL weights are not provided."
        assert len(irl_weights) == len(features), "IRL weights and features have different lengths."
        # Compute the reward
        if info["collision"] or info["off_road"]:
            return -np.inf
        return np.dot(irl_weights, features)

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


"""
Define the reward function for the environment. The reward function is a callable that takes the following arguments:

        state: The current state of the environment.
        action: The action taken by the agent.
        next_state: The next state of the environment.
        done: A boolean indicating whether the episode has ended.
        info: A dictionary containing any additional information.
"""


def get_reward(terminated, truncated, info, use_rl_reward=False):
    """
    Reward function for the environment.
    Currently, it is a sparse reward function that is zero but when agent reaches the goal, collides or goes off-road.
    """

    if use_rl_reward:
        # Load the weights
        # We need to compute these features
        # ego_speed, abs(ego_long_acc), abs(ego_lat_acc), abs(ego_long_jerk),
        #                              thw_front, thw_rear, collision, social_impact
        raise NotImplementedError
        features = compute_features(state, action, next_state, done, info)
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
            return 0  # TODO: add self._time_discount ** trajectory.duration


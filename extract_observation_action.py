import json
from dataclasses import dataclass
import os
from typing import Optional, List
import numpy as np
import pandas as pd
import pickle

from sim4ad.path_utils import write_common_property
from sim4ad.common_constants import MISSING_NEARBY_AGENT_VALUE


@dataclass
class SurroundingAgentFeature:
    rel_dx: Optional[float] = None
    rel_dy: Optional[float] = None
    vx: Optional[float] = None
    vy: Optional[float] = None
    ax: Optional[float] = None
    ay: Optional[float] = None
    psi: Optional[float] = None


class ExtractObservationAction:
    """
    Extract observation and action from the clustered data
    """

    def __init__(self, episodes, clustered_dataframe):
        """
        Args:
            episodes: the episodes from the original dataset
            clustered_dataframe: clustered trajectories
        """
        self._clustered_dataframe = clustered_dataframe
        self._episodes = episodes
        self._demonstrations = {"observations": [], "actions": []}

        # Create a dataframe with the columns of the demonstrations list
        self._demonstrations_df = pd.DataFrame(columns=['observations', 'actions'])

        # save for IRL training
        self._irl_data = {}

    def extract_demonstrations(self):
        """Extract observations"""
        def combine(x, y):
            return np.sqrt(np.array(x) ** 2 + np.array(y) ** 2)

        grouped_cluster = self._clustered_dataframe.groupby('label')
        # we train one driving style as an example
        # TODO: Change for different driving styles
        driving_style_a = grouped_cluster.get_group(0)

        for episode in self._episodes:
            episode_id = episode.config.recording_id
            matching_rows = driving_style_a.iloc[:, 0] == episode_id
            if matching_rows.any():
                agent_ids = driving_style_a[matching_rows]['agent_id']
                for agent_id in agent_ids:
                    agent = episode.agents[agent_id]

                    self._irl_data[agent_id] = agent

                    # calculate steering angle
                    delta = self.extract_steering_angle(agent)

                    speed = combine(agent.vx_vec, agent.vy_vec)
                    ego_agent_observations = {'time': agent.time, 'speed': speed,
                                              'heading': agent.psi_vec,  # 'aid': agent_id, TODO: include it!
                                              # 'eid': episode_id,
                                              'distance_left_lane_marking': agent.distance_left_lane_marking,
                                              'distance_right_lane_marking': agent.distance_right_lane_marking}

                    # todo: explain that ego_agent_observations should be a dataframe where the rows are the timestamps
                    # and the columns are the features. then, we have a list of these dataframes, one for each agent

                    acceleration = combine(agent.ax_vec, agent.ay_vec)
                    ego_agent_actions = {'acceleration': acceleration, 'steer_angle': delta}

                    for inx, t in enumerate(agent.time):
                        # get surrounding agent's information
                        surrounding_agents = agent.object_relation_dict_list[inx]
                        for surrounding_agent_relation, surrounding_agent_id in surrounding_agents.items():
                            if surrounding_agent_id is not None:
                                surrounding_agent = episode.agents[surrounding_agent_id]
                                # todo lat_dist_dict_vec, long_dist_dict_vec are none, so should be recalculated
                                long_distance, lat_distance = agent.get_lat_and_long(t, surrounding_agent)
                                surrounding_agent_inx = surrounding_agent.next_index_of_specific_time(t)
                                surrounding_rel_dx = long_distance
                                surrounding_rel_dy = lat_distance
                                surrounding_v = combine(surrounding_agent.vx_vec[surrounding_agent_inx],
                                                        surrounding_agent.vy_vec[surrounding_agent_inx])
                                surrounding_a = combine(surrounding_agent.ax_vec[surrounding_agent_inx],
                                                        surrounding_agent.ay_vec[surrounding_agent_inx])
                                surrounding_heading = surrounding_agent.psi_vec[surrounding_agent_inx]
                            else:
                                # Set to invalid value if there is no surrounding agent
                                surrounding_rel_dx = surrounding_rel_dy = surrounding_v = surrounding_a \
                                    = surrounding_heading = MISSING_NEARBY_AGENT_VALUE

                            # if the surrounding agent is not in the dictionary, add it
                            if surrounding_agent_relation not in ego_agent_observations:
                                ego_agent_observations[f'{surrounding_agent_relation}_rel_dx'] = []
                                ego_agent_observations[f'{surrounding_agent_relation}_rel_dy'] = []
                                ego_agent_observations[f'{surrounding_agent_relation}_v'] = []
                                ego_agent_observations[f'{surrounding_agent_relation}_a'] = []
                                ego_agent_observations[f'{surrounding_agent_relation}_heading'] = []

                            ego_agent_observations[f'{surrounding_agent_relation}_rel_dx'].append(surrounding_rel_dx)
                            ego_agent_observations[f'{surrounding_agent_relation}_rel_dy'].append(surrounding_rel_dy)
                            ego_agent_observations[f'{surrounding_agent_relation}_v'].append(surrounding_v)
                            ego_agent_observations[f'{surrounding_agent_relation}_a'].append(surrounding_a)
                            ego_agent_observations[f'{surrounding_agent_relation}_heading'].append(surrounding_heading)

                    ego_agent_observations = pd.DataFrame(ego_agent_observations, index=agent.time)

                    # todo: if we want to export the dataset, keep it as is - otherwise only get the values without
                    # the column names and drop time
                    ego_agent_observations = ego_agent_observations.drop(columns=['time'])
                    ego_agent_actions = pd.DataFrame(ego_agent_actions, index=agent.time)

                    # Update the features used in the observations in the file that keeps track of the common
                    # part across the project, common_elements.json.
                    write_common_property('FEATURES_IN_OBSERVATIONS', ego_agent_observations.columns.tolist())
                    write_common_property('FEATURES_IN_ACTIONS', ego_agent_actions.columns.tolist())

                    ego_agent_observations = ego_agent_observations.values
                    ego_agent_actions = ego_agent_actions.values

                    self._demonstrations["observations"].append(ego_agent_observations)
                    self._demonstrations["actions"].append(ego_agent_actions)

    @staticmethod
    def extract_steering_angle(agent) -> List:
        """Extract steering_angle here"""
        dt = agent.delta_t
        theta_dot_vec = [(agent.psi_vec[i] - agent.psi_vec[i - 1]) / dt for i in range(1, len(agent.psi_vec))]
        # make sure yaw_rate has the same length as time
        theta_dot_vec.insert(0, theta_dot_vec[0])

        # approximate the wheelbase using a vehicle's length (could occur errors)
        wheel_base = agent.length * 0.6

        # TODO: need to be verified
        deltas = []
        for inx, theta_dot in enumerate(theta_dot_vec):
            v = np.sqrt(agent.vx_vec[inx] ** 2 + agent.vy_vec[inx] ** 2)
            delta_t = np.arctan2(theta_dot * wheel_base, v)
            deltas.append(delta_t)

        return deltas

    def save_trajectory(self):
        """Save a list of trajectories, and each trajectory include (state, action) pair"""
        folder_path = 'scenarios/data/trainingdata'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        for episode in self._episodes:
            episode_id = episode.config.recording_id
            episode_path = os.path.join(folder_path, episode_id)
            if not os.path.exists(episode_path):
                os.makedirs(episode_path)

            # Saving the demonstration data to a file
            with open(episode_path + "/demonstration.pkl", "wb") as file:
                pickle.dump(self._demonstrations, file)

            # Saving IRL data to a file
            with open(episode_path + "/irl.pkl", "wb") as file:
                pickle.dump(self._irl_data, file)

from collections import defaultdict
from dataclasses import dataclass
import os
from typing import Optional, List
import numpy as np
import pandas as pd
import pickle
import joblib
from feature_normalization import extract_features
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


class MDPValues:
    def __init__(self):
        self.observations: Optional[List] = []
        self.actions: Optional[List] = []
        self.rewards: Optional[List] = []
        self.terminals: Optional[List] = []


class ExtractObservationAction:
    """
    Extract observation and action from the clustered data
    """

    def __init__(self, split, map_name, episodes, driving_style):
        """
        Args:
            split: represents if the data belongs to training, testing or validation
            map_name: the name of the map
            episodes: the episodes from the original dataset
            driving_style: the driving style used for computing the rewards
        """
        self._split = split
        self._map_name = map_name
        self._driving_style = driving_style
        self._episodes = episodes

        self._theta = self.load_reward_weights()
        self._feature_mean_std = self.load_feature_normalization()
        self._clustered_demonstrations = {"General": [], "clustered": []}

    @staticmethod
    def combine(x, y):
        return np.sqrt(np.array(x) ** 2 + np.array(y) ** 2)

    def extract_mdp(self, episode, aid, agent):
        """Extract mdp values for one agent"""
        mdp = MDPValues()
        # calculate yaw rate
        yaw_rate = self.extract_yaw_rate(agent)

        speed = self.combine(agent.vx_vec, agent.vy_vec)

        ego_agent_observations = defaultdict(list)
        ego_agent_observations['time'] = agent.time
        ego_agent_observations['speed'] = speed
        ego_agent_observations['heading'] = agent.psi_vec
        ego_agent_observations['distance_left_lane_marking'] = agent.distance_left_lane_marking
        ego_agent_observations['distance_right_lane_marking'] = agent.distance_right_lane_marking

        # todo: explain that ego_agent_observations should be a dataframe where the rows are the timestamps
        # and the columns are the features. then, we have a list of these dataframes, one for each agent

        acceleration = self.combine(agent.ax_vec, agent.ay_vec)
        ego_agent_actions = {'acceleration': acceleration, 'yaw_rate': yaw_rate}
        ego_agent_features = []

        skip_vehicle = False
        for inx, t in enumerate(agent.time):
            # get surrounding agent's information
            try:
                surrounding_agents = agent.object_relation_dict_list[inx]
            except IndexError as e:
                print(f"KeyError: {e}. Skipping vehicle {aid}.")
                skip_vehicle = True
                break

            for surrounding_agent_relation, surrounding_agent_id in surrounding_agents.items():
                if surrounding_agent_id is not None:
                    surrounding_agent = episode.agents[surrounding_agent_id]
                    # lat_dist_dict_vec, long_dist_dict_vec are none, so should be recalculated using get_lat_and_long
                    long_distance, lat_distance = agent.get_lat_and_long(t, surrounding_agent)
                    surrounding_agent_inx = surrounding_agent.next_index_of_specific_time(t)
                    surrounding_rel_dx = long_distance
                    surrounding_rel_dy = lat_distance
                    surrounding_rel_speed = (self.combine(surrounding_agent.vx_vec[surrounding_agent_inx],
                                                          surrounding_agent.vy_vec[surrounding_agent_inx])
                                             - speed[inx])
                    surrounding_rel_a = (self.combine(surrounding_agent.ax_vec[surrounding_agent_inx],
                                                      surrounding_agent.ay_vec[surrounding_agent_inx])
                                         - acceleration[inx])
                    surrounding_heading = surrounding_agent.psi_vec[surrounding_agent_inx]

                else:
                    # Set to invalid value if there is no surrounding agent
                    surrounding_rel_dx = surrounding_rel_dy = surrounding_rel_speed = surrounding_rel_a \
                        = surrounding_heading = MISSING_NEARBY_AGENT_VALUE

                ego_agent_observations[f'{surrounding_agent_relation}_rel_dx'].append(surrounding_rel_dx)
                ego_agent_observations[f'{surrounding_agent_relation}_rel_dy'].append(surrounding_rel_dy)
                ego_agent_observations[f'{surrounding_agent_relation}_rel_speed'].append(surrounding_rel_speed)
                ego_agent_observations[f'{surrounding_agent_relation}_rel_a'].append(surrounding_rel_a)
                ego_agent_observations[f'{surrounding_agent_relation}_heading'].append(surrounding_heading)

            # extract features to compute rewards
            features = extract_features(inx, t, agent, episode)

            # normalize features
            normalized_features = self.exponential_normalization(features)

            ego_agent_features.append(normalized_features)

        if not skip_vehicle:
            ego_agent_observations = pd.DataFrame(ego_agent_observations, index=agent.time)

            # todo: if we want to export the dataset, keep it as is - otherwise only get the values without
            # the column names and drop time
            ego_agent_observations = ego_agent_observations.drop(columns=['time'])
            ego_agent_actions = pd.DataFrame(ego_agent_actions, index=agent.time)

            # Update the features used in the observations in the file that keeps track of the common
            # part across the project, common_elements.json.
            write_common_property('FEATURES_IN_OBSERVATIONS', ego_agent_observations.columns.tolist())
            write_common_property('FEATURES_IN_ACTIONS', ego_agent_actions.columns.tolist())

            # save mdp values
            mdp.observations = ego_agent_observations.values
            mdp.actions = ego_agent_actions.values
            mdp.rewards = [np.dot(feature, self._theta) for feature in ego_agent_features]
            mdp.terminals = [False for _ in range(len(agent.time)-1)] + [True]

            return mdp
        else:
            return None

    def zscore_normalization(self, features):
        """Using Z-Score to normalize the features"""
        features = np.array(features)
        mean = self._feature_mean_std['mean']
        std = self._feature_mean_std['std']
        std_safe = np.where(std == 0, 1, std)  # Replace 0s with 1s in std array to avoid division by zero
        normalized_features = (features - mean) / std_safe

        return normalized_features

    @staticmethod
    def exponential_normalization(features):
        """Using exponential for normalization"""
        normalized_features = [None for _ in range(len(features))]
        for inx, feature in enumerate(features):
            # skip THW, collision
            if inx == 4 or 5 or 6:
                normalized_features[inx] = feature
            else:
                normalized_features[inx] = np.exp(-1/feature)

        return normalized_features

    def extract_demonstrations(self):
        """Extract observations"""
        key = 'General'
        if self._driving_style != 'General':
            key = 'clustered'

        for episode in self._episodes:
            for aid, agent in episode.agents.items():
                agent_mdp_values = self.extract_mdp(episode, aid, agent)
                if agent_mdp_values is None:
                    continue

                self._clustered_demonstrations[key].append(agent_mdp_values)

        self.save_trajectory()

    def load_reward_weights(self):
        """Loading reward weights theta derived from IRL"""
        with open('results/' + self._driving_style + 'training_log.pkl', 'rb') as file:
            data = pickle.load(file)
        return data['theta'][-1]

    @staticmethod
    def load_feature_normalization():
        """Loading the mean and standard deviation for feature normalization"""
        return joblib.load('results/feature_normalization.pkl')

    @staticmethod
    def extract_yaw_rate(agent) -> List:
        """Extract yaw rate here"""
        dt = agent.delta_t
        theta_dot_vec = [(agent.psi_vec[i] - agent.psi_vec[i - 1]) / dt for i in range(1, len(agent.psi_vec))]
        # make sure yaw_rate has the same length as time
        theta_dot_vec.insert(0, theta_dot_vec[0])

        return theta_dot_vec

    def save_trajectory(self):
        """Save a list of trajectories, and each trajectory include (state, action) pair"""
        folder_path = 'scenarios/data/' + self._split
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Saving the demonstration data to a file
        with open(folder_path + "/" + self._driving_style + self._map_name + "_demonstration.pkl", "wb") as file:
            pickle.dump(self._clustered_demonstrations, file)
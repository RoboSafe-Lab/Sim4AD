from collections import defaultdict
from dataclasses import dataclass
import os
from typing import Optional, List
import numpy as np
import pandas as pd
import pickle
import joblib
from feature_normalization import extract_features
from sim4ad.opendrive import Map, plot_map
from sim4ad.path_utils import write_common_property
from sim4ad.common_constants import MISSING_NEARBY_AGENT_VALUE, REMOVED_AGENTS
from sim4ad.irlenv.irlenvsim import IRLEnv
from loguru import logger
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

@dataclass
class SurroundingAgentFeature:
    rel_dx: Optional[float] = None
    rel_dy: Optional[float] = None
    velocity: Optional[float] = None
    acceleration: Optional[float] = None
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
        # no longer using z-score
        # self._feature_mean_std = self.load_feature_normalization()
        self._clustered_demonstrations = {"All": [], "clustered": []}

    def extract_mdp(self, episode, aid, agent):
        """Extract mdp values for one agent"""
        mdp = MDPValues()
        # calculate yaw rate
        yaw_rate = self.extract_yaw_rate(agent)

        ego_agent_observations = defaultdict(list)
        ego_agent_observations['time'] = agent.time
        ego_agent_observations['velocity'] = agent.vx_vec
        ego_agent_observations['cos_heading'] = np.cos(agent.psi_vec)
        ego_agent_observations['sin_heading'] = np.sin(agent.psi_vec)
        ego_agent_observations['distance_left_lane_marking'] = agent.distance_left_lane_marking
        ego_agent_observations['distance_right_lane_marking'] = agent.distance_right_lane_marking

        # todo: explain that ego_agent_observations should be a dataframe where the rows are the timestamps
        # and the columns are the features. then, we have a list of these dataframes, one for each agent

        # The acceleration is the longitudinal acceleration, hence the acceleration in the direction of the heading.
        ego_agent_actions = {'acceleration': agent.ax_vec, 'yaw_rate': yaw_rate}
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
                    surrounding_rel_vx = (surrounding_agent.vx_vec[surrounding_agent_inx] - agent.vx_vec[inx])
                    surrounding_rel_ax = (surrounding_agent.ax_vec[surrounding_agent_inx] - agent.ax_vec[inx])
                    surrounding_heading = surrounding_agent.psi_vec[surrounding_agent_inx]

                else:
                    # Set to invalid value if there is no surrounding agent
                    surrounding_rel_dx = surrounding_rel_dy = surrounding_rel_vx \
                        = surrounding_rel_ax = surrounding_heading = MISSING_NEARBY_AGENT_VALUE

                ego_agent_observations[f'{surrounding_agent_relation}_rel_dx'].append(surrounding_rel_dx)
                ego_agent_observations[f'{surrounding_agent_relation}_rel_dy'].append(surrounding_rel_dy)
                ego_agent_observations[f'{surrounding_agent_relation}_rel_v'].append(surrounding_rel_vx)
                # ax is the longitudinal acceleration, hence the acceleration in the direction of the heading.
                # we don't need ay here, as it is the lateral acceleration
                ego_agent_observations[f'{surrounding_agent_relation}_rel_a'].append(surrounding_rel_ax)
                ego_agent_observations[f'{surrounding_agent_relation}_cos_heading'].append(
                    np.cos(surrounding_heading) if surrounding_heading != MISSING_NEARBY_AGENT_VALUE else MISSING_NEARBY_AGENT_VALUE)
                ego_agent_observations[f'{surrounding_agent_relation}_sin_heading'].append(
                    np.sin(surrounding_heading) if surrounding_heading != MISSING_NEARBY_AGENT_VALUE else MISSING_NEARBY_AGENT_VALUE)

            # extract features to compute rewards
            features = extract_features(inx, t, agent, episode)
            irl = IRLEnv()
            normalized_features = irl.feature_normalization(features)
            ego_agent_features.append(normalized_features)

            # scenario_map = Map.parse_from_opendrive(episode.map_file)
            # plot_map(scenario_map, markings=True, midline=False, drivable=True, plot_background=False)
            # plt.scatter(agent.x_vec[:inx], agent.y_vec[:inx])
            # print(f'weights are: {self._theta}')
            # print(f'features are: {normalized_features}')
            #
            # mean = 7.043490768410265e-07
            # std =0.9999998807907104
            # reward = np.dot(normalized_features, self._theta)
            # normalized_reward = (reward - mean) / std
            # print(f'reward is {reward}, normalized reward is {normalized_reward}')
            # plt.text(agent.x_vec[inx], agent.y_vec[inx] + 0.2, f'R: {(np.dot(normalized_features, self._theta) - mean)/std:.2f}', fontsize=8, ha='center',
            #          bbox=dict(boxstyle='round,pad=0.2', facecolor='wheat', alpha=0.5))
            # plt.show()

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
            heading_inx = []
            for inx, key in enumerate(ego_agent_observations.keys()):
                if 'heading' in key:
                    heading_inx.append(inx)
            write_common_property('HEADING_IN_FEATURES', heading_inx)

            # save mdp values
            mdp.observations = ego_agent_observations.values
            mdp.actions = ego_agent_actions.values
            mdp.rewards = [np.dot(feature, self._theta) for feature in ego_agent_features]
            mdp.terminals = [False for _ in range(len(agent.time) - 1)] + [True]

            return mdp
        else:
            return None

    @staticmethod
    def zscore_normalization(features, mean: float, std: float):
        """Using Z-Score to normalize the features"""
        features = np.array(features)
        std_safe = np.where(std == 0, 1, std)  # Replace 0s with 1s in std array to avoid division by zero
        normalized_features = (features - mean) / std_safe

        return normalized_features

    def extract_demonstrations(self):
        """Extract observations"""
        key = 'All'
        if self._driving_style != 'All':
            key = 'clustered'

        for episode in self._episodes:
            for aid, agent in episode.agents.items():
                if aid in REMOVED_AGENTS:
                    # # print the position and the map
                    # print(aid)
                    # map = Map.parse_from_opendrive(episode.map_file)
                    # import matplotlib.pyplot as plt
                    # fig, ax = plt.subplots()
                    # plot_map(map, markings=True,
                    #          hide_road_bounds_in_junction=True, ax=ax)
                    # plt.scatter(agent.x_vec, agent.y_vec)
                    # plt.show()
                    logger.info(f'{aid} is removed because of inaccurate data')
                    continue  # known issue with this agent, where it spawns out of the road, agent with sudden
                    # changed velocity
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
        theta_dot_vec = [(agent.psi_vec[i + 1] - agent.psi_vec[i]) / dt for i in range(0, len(agent.psi_vec) - 1)]
        # make sure yaw_rate has the same length as time
        theta_dot_vec.insert(len(agent.psi_vec), theta_dot_vec[-1])

        return theta_dot_vec

    def save_trajectory(self):
        """Save a list of trajectories, and each trajectory include (state, action) pair"""
        folder_path = 'scenarios/data/' + self._split
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Saving the demonstration data to a file
        with open(folder_path + "/" + self._driving_style + self._map_name + "_demonstration.pkl", "wb") as file:
            pickle.dump(self._clustered_demonstrations, file)

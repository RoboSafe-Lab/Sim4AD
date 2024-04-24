from collections import defaultdict
from dataclasses import dataclass
import os
from typing import Optional, List
import numpy as np
import pandas as pd
import pickle

from sim4ad.path_utils import write_common_property
from sim4ad.common_constants import MISSING_NEARBY_AGENT_VALUE
from sim4ad.irlenv.vehicle.behavior import IDMVehicle


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
        self._clustered_demonstrations = {"General": MDPValues(), "clustered": MDPValues()}

    @staticmethod
    def combine(x, y):
        return np.sqrt(np.array(x) ** 2 + np.array(y) ** 2)

    @staticmethod
    def desired_gap(ego_v: float, ego_length: float, rear_agent_v: float, rear_agent_length: float):
        """
        Compute the desired distance between a vehicle and its leading vehicle.

        :param ego_v: the velocity of the ego
        :param ego_length: length
        :param rear_agent_v: the velocity of the rear agent
        :param rear_agent_length: length
        :return: the desired distance between the two vehicles
        """
        d0 = IDMVehicle.DISTANCE_WANTED + ego_length / 2 + rear_agent_length / 2
        tau = IDMVehicle.TIME_WANTED
        ab = -IDMVehicle.COMFORT_ACC_MAX * IDMVehicle.COMFORT_ACC_MIN
        dv = rear_agent_v - ego_v
        d_star = d0 + rear_agent_v * tau + rear_agent_v * dv / (2 * np.sqrt(ab))

        return d_star

    @staticmethod
    def extract_features(inx, agent) -> List:
        """Using the same features from inverse RL"""
        # travel efficiency
        ego_speed = np.exp(-1/abs(agent.vx_vec[inx])) if agent.vx_vec[inx] != 0 else 0

        # comfort
        ego_long_acc = np.exp(-abs(agent.ax_vec[inx]))
        ego_lat_acc = np.exp(-abs(agent.ay_vec[inx]))
        if agent.jerk_x_vec is None:
            ego_long_jerk = np.exp(-abs((agent.ax_vec[inx] - agent.ax_vec[inx - 1]) / agent.delta_t)) if inx > 0 else 0
        else:
            ego_long_jerk = np.exp(-abs(agent.jerk_x_vec[inx]))

        # time headway front (thw_front) and time headway behind (thw_rear)
        thw_front = agent.tth_dict_vec[inx]['front_ego']
        thw_rear = agent.tth_dict_vec[inx]['behind_ego']
        thw_front = np.exp(-1 / thw_front) if thw_front is not None else 1
        thw_rear = np.exp(-1 / thw_rear) if thw_rear is not None else 1

        # no collision in the dataset
        collision = -1

        # feature array
        features = [ego_speed, ego_long_acc, ego_lat_acc, ego_long_jerk, thw_front, thw_rear, collision]

        return features

    def extract_mdp(self, episode, aid, agent):
        """Extract mdp values for one agent"""
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
            social_impact = 0
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
                    # todo lat_dist_dict_vec, long_dist_dict_vec are none, so should be recalculated
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

                    # for computing social impact
                    if surrounding_agent_relation == 'behind_ego':
                        desired_gap = self.desired_gap(agent.vx_vec[inx], agent.length,
                                                       surrounding_agent.vx_vec[surrounding_agent_inx],
                                                       surrounding_agent.length)
                        if long_distance < desired_gap and surrounding_agent.ax_vec[surrounding_agent_inx] < 0:
                            social_impact = surrounding_agent.ax_vec[surrounding_agent_inx]

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
            features = self.extract_features(inx, agent)
            features.append(social_impact)
            ego_agent_features.append(features)

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

            observations = ego_agent_observations.values
            actions = ego_agent_actions.values
            rewards = [np.dot(feature, self._theta) for feature in ego_agent_features]
            terminals = [False for _ in range(len(agent.time)-1)] + [True]

            return observations, actions, rewards, terminals
        else:
            return None

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

                self._clustered_demonstrations[key].observations.append(agent_mdp_values[0])
                self._clustered_demonstrations[key].actions.append(agent_mdp_values[1])
                self._clustered_demonstrations[key].rewards.append(agent_mdp_values[2])
                self._clustered_demonstrations[key].terminals.append(agent_mdp_values[3])

        self.save_trajectory()

    def load_reward_weights(self):
        """Loading reward weights theta derived from IRL"""
        with open('results/' + self._driving_style + 'training_log.pkl', 'rb') as file:
            data = pickle.load(file)
        return data['theta'][-1]

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
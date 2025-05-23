from sim4ad.util import parse_args, load_dataset
from sim4ad.path_utils import get_config_path
import numpy as np
import joblib
import sys
from typing import List, Optional
from sim4ad.irlenv.vehicle.behavior import IDMVehicle
from simulator.simulator_util import compute_distance_markings
from sim4ad.opendrive import Map
from sim4ad.irlenv.irlenvsim import IRLEnv
from loguru import logger


def save_normalization(agents_features):
    """Save the mean and std of feature values for Z-Score normalization"""
    mean = np.mean(agents_features, axis=0)
    std = np.std(agents_features, axis=0)
    # Save the mean and std to a file using joblib
    joblib.dump({'mean': mean, 'std': std}, 'results/feature_normalization.pkl')


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


def extract_features(inx, t, agent, episode) -> Optional[List]:
    """Using the same features from inverse RL"""
    # travel efficiency
    ego_speed = abs(agent.vx_vec[inx])

    # comfort
    ego_long_acc = abs(agent.ax_vec[inx])
    ego_lat_acc = abs(agent.ay_vec[inx])
    if agent.jerk_x_vec is None:
        ego_long_jerk = (agent.ax_vec[inx] - agent.ax_vec[inx - 1]) / agent.delta_t if inx > 0 else 0
    else:
        ego_long_jerk = agent.jerk_x_vec[inx]
    ego_long_jerk = abs(ego_long_jerk)

    # time headway front (thw_front) and time headway behind (thw_rear)
    try:
        thw_front = agent.tth_dict_vec[inx]['front_ego']
    except IndexError as e:
        return None
    thw_rear = agent.tth_dict_vec[inx]['behind_ego']
    thw_front =  thw_front if thw_front is not None else np.inf
    thw_rear = thw_rear if thw_rear is not None else np.inf
    thw_front = np.exp(-1/thw_front)
    thw_rear = np.exp(-1/thw_rear)

    # centerline deviation
    d = (agent.distance_left_lane_marking[inx] - agent.distance_right_lane_marking[inx]) / 2
    d_centerline = abs(d)

    lane_deviation_rate = 0.0
    #if len(agent.distance_left_lane_marking) > 1:
    lane = agent.lane_id_vec[inx]
    if inx >0:
        lane_previous = agent.lane_id_vec[inx-1]
        if lane_previous == lane :
            d = (agent.distance_left_lane_marking[inx - 1] - agent.distance_right_lane_marking[inx - 1]) / 2
            d_centerline_previous = abs(d)
            lane_deviation_rate = abs(d_centerline - d_centerline_previous) / agent.delta_t
        else:
            d = (agent.distance_left_lane_marking[inx - 1] - agent.distance_right_lane_marking[inx - 1]) / 2
            d_centerline_previous = abs(d)
            d_lane_previous = (agent.distance_left_lane_marking[inx - 1] + agent.distance_right_lane_marking[inx - 1]) / 2
            d_lane = (agent.distance_left_lane_marking[inx] + agent.distance_right_lane_marking[inx]) / 2
            lane_deviation_rate = (d_lane_previous - d_centerline_previous + d_lane - d_centerline) / agent.delta_t

    # nearest_distance_lane_marking = min(abs(agent.distance_left_lane_marking[inx]),
    #                                     abs(agent.distance_right_lane_marking[inx]))

    # lane availability features
    center = np.array([float(agent.x_vec[inx]), float(agent.y_vec[inx])])
    heading = agent.psi_vec[inx]
    #scenario_map = Map.parse_from_opendrive(episode.map_file)
    #lane = scenario_map.best_lane_at(center, heading)
    #left_lane_available, right_lane_available = IRLEnv.check_adjacent_lanes(lane)
    #lane = agent.lane_id_vec[inx]
    if lane == 2 or lane == -2:
        left_lane_available = False
        right_lane_available = True
    elif lane == 4 or lane == -4:
        left_lane_available = True
        right_lane_available = False   
    else:
        left_lane_available = True
        right_lane_available = True   

    # feature array
    features = [ego_speed, ego_long_acc, ego_lat_acc, ego_long_jerk, thw_front, thw_rear, d_centerline,
                lane_deviation_rate, left_lane_available, right_lane_available]

    return features


def main():
    args = parse_args()
    data = load_dataset(get_config_path(args.map))

    episodes = data.scenario.episodes

    agents_features = []
    for episode in episodes:
        for aid, agent in episode.agents.items():
            for inx, t in enumerate(agent.time):
                features = extract_features(inx=inx, t=t, agent=agent, episode=episode)
                if features is not None:
                    agents_features.append(features)

    # save mean and std for online RL and inverse RL feature normalization
    save_normalization(agents_features=agents_features)
    logger.info('Save mean and std for feature normalization.')


if __name__ == '__main__':
    sys.exit(main())

from typing import List
from loguru import logger
import numpy as np
import pickle

from sim4ad.data.data_loaders import DatasetDataLoader
from sim4ad.util import parse_args
from sim4ad.path_utils import get_config_path
from sim4ad.opendrive import Map
from sim4ad.irlenv import IRLEnv, IRL


def load_dataset(config_path: str = None, evaluation_data: List[str] = None):
    """Loading clustered trajectories"""
    data_loader = DatasetDataLoader(config_path, evaluation_data)
    data_loader.load()

    return data_loader


def load_max_feature():
    """Load the maximum feature values for training"""
    with open('max_feature.txt', 'r') as f:
        max_feature = f.read().splitlines()
    return max_feature


def load_theta():
    """Load the optimized theta from IRL"""
    with open('training_log.pkl', 'rb') as f:
        training_log = pickle.load(f)

    return training_log['theta'][-1]


class IRLEva(IRLEnv):
    def __init__(self, episode, scenario_map, ego, IDM):
        self._max_feature = load_max_feature()
        self._theta = load_theta()
        super().__init__(episode, scenario_map, ego, IDM)

    def get_trajectory_one_timestep(self, time):
        """generate the trajectory for one agent"""

        logger.info(f"Simulation time: {time}")

        self.reset(reset_time=time)

        buffer_scene = self.get_buffer_scene(time)

        # Normalize the feature
        for traj in buffer_scene:
            for i in range(IRL.feature_num):
                if self._max_feature[i] == 0:
                    traj[2][i] = 0
                else:
                    traj[2][i] /= self._max_feature[i]

        # evaluate trajectories
        reward_hl = []
        for trajectory in buffer_scene:
            reward = np.dot(trajectory[2], self._theta)
            reward_hl.append([reward, trajectory[3]])  # reward, human likeness

        # calculate probability of each trajectory
        rewards = [traj[0] for traj in reward_hl]
        probs = [np.exp(reward) for reward in rewards]
        probs = probs / np.sum(probs)

        # select trajectories to calculate human likeness
        idx = probs.argsort()[-3:][::-1]
        hl = np.min([reward_hl[i][-1] for i in idx])

        # TODO: return the trajectory
        pass

    def simulation(self):
        """based on the optimal trajectory to update each agent's state"""
        pass


def main():
    args = parse_args()

    # we do the evaluation here
    test_data = load_dataset(get_config_path(args.map), ['test'])

    for episode in test_data.scenario.episodes:
        scenario_map = Map.parse_from_opendrive(episode.map_file)
        for frame in episode.frames:
            # for each agent generate trajectory using the weights from IRL.
            for aid, agent_state in frame.agents.items():
                logger.info(f"Ego agent: {aid}")
                agent = episode.agents[aid]
                irl_eva = IRLEva(episode=episode, scenario_map=scenario_map, ego=agent, IDM=False)
                irl_eva.get_trajectory_one_timestep(frame.time)


if __name__ == "__main__":
    main()

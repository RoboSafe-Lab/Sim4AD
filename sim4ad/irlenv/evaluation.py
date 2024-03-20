from typing import List
from loguru import logger
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sim4ad.data.data_loaders import DatasetDataLoader
from sim4ad.util import parse_args
from sim4ad.path_utils import get_config_path
from sim4ad.opendrive import Map
from sim4ad.irlenv import IRLEnv, IRL
from sim4ad.opendrive import plot_map


def load_dataset(config_path: str = None, evaluation_data: List[str] = None):
    """Loading clustered trajectories"""
    data_loader = DatasetDataLoader(config_path, evaluation_data)
    data_loader.load()

    return data_loader


def load_max_feature():
    """Load the maximum feature values for training"""
    with open('results/max_feature.txt', 'r') as f:
        max_feature = [float(line) for line in f.read().splitlines()]
    return max_feature


def load_theta():
    """Load the optimized theta from IRL"""
    with open('results/training_log.pkl', 'rb') as f:
        training_log = pickle.load(f)

    return training_log['theta'][-1]


class IRLEva(IRLEnv):
    def __init__(self, episode, scenario_map):
        self._max_feature = load_max_feature()
        self._theta = load_theta()
        super().__init__(episode, scenario_map)

    def get_trajectory_one_timestep(self, time):
        """generate the trajectory for one agent"""
        self.reset(reset_time=time)

        buffer_scene = self.get_buffer_scene(time, save_planned_tra=True)

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
        # find the indices of the largest 3 values
        idx = probs.argsort()[-3:][::-1]
        hl = np.min([reward_hl[i][-1] for i in idx])

        # the first value of idx indicates the highest probability
        tra = buffer_scene[idx[0]]

        return tra


class CreateAnimation:
    def __init__(self, scenario_map):
        self.fig, self.ax = plt.subplots()
        plot_map(scenario_map, ax=self.ax, markings=True, midline=False, drivable=True,
                 plot_background=False)
        self.vehicles = {}

    def update_and_show(self, agents_states, time, delta_t):
        # Update each vehicle's trajectory based on the current states
        for vehicle_id, agent_states in agents_states.items():
            tra = np.array(agent_states[1][0])  # Unpack positions into x and y coordinates

            if vehicle_id not in self.vehicles:
                # If this vehicle hasn't been plotted yet, create a Line2D object for it
                self.vehicles[vehicle_id] = Line2D([], [], linestyle='-', marker='o')
                self.ax.add_line(self.vehicles[vehicle_id])

            # plot current position
            inx = min(round((time - agent_states[0]) / delta_t), len(tra) - 1)
            self.vehicles[vehicle_id].set_data([tra[inx, 0]], [tra[inx, 1]])

        self.ax.set_title(f'T = {time}')
        plt.draw()
        plt.pause(0.01)


def main():
    args = parse_args()

    # we do the evaluation here
    test_data = load_dataset(get_config_path(args.map), ['test'])

    for episode in test_data.scenario.episodes:
        scenario_map = Map.parse_from_opendrive(episode.map_file)
        animation = CreateAnimation(scenario_map)
        agents_states = {}
        irl_eva = IRLEva(episode=episode, scenario_map=scenario_map)

        # define the trajectory replan frequency, counted by num * delta_t
        replan_frequency = 1 * irl_eva.delta_t
        replan_frequency = min(replan_frequency, irl_eva.forward_simulation_time)
        for frame in episode.frames:
            # for each agent generate trajectory using the weights from IRL.
            for aid, agent_state in frame.agents.items():
                if aid in agents_states and frame.time - agents_states[aid][0] <= replan_frequency:
                    continue

                # TODO: if replan frequency is too large, replan occurs a discrepancy
                agent = episode.agents[aid]
                irl_eva.ego = agent

                tra = irl_eva.get_trajectory_one_timestep(frame.time)
                agents_states[aid] = (frame.time, tra)

            animation.update_and_show(agents_states, frame.time, irl_eva.delta_t)
            pass


if __name__ == "__main__":
    main()

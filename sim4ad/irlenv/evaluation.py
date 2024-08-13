from typing import List
from dataclasses import dataclass
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import imageio
from loguru import logger

from sim4ad.data.data_loaders import DatasetDataLoader
from sim4ad.util import parse_args
from sim4ad.path_utils import get_config_path
from sim4ad.opendrive import Map
from sim4ad.irlenv import IRLEnv
from sim4ad.opendrive import plot_map
from sim4ad.agentstate import AgentState


@dataclass
class AgentReset:
    state: AgentState
    trj: None


def load_dataset(config_path: str = None, evaluation_data: List[str] = None):
    """Loading clustered trajectories"""
    data_loader = DatasetDataLoader(config_path, evaluation_data)
    data_loader.load()

    return data_loader


def load_theta(driving_style: str):
    """Load the optimized theta from IRL"""
    with open('results/' + driving_style + 'training_log.pkl', 'rb') as f:
        training_log = pickle.load(f)

    return training_log['theta'][-1]


class IRLEva(IRLEnv):
    def __init__(self, episode, scenario_map, ego, driving_style: str):
        self._theta = load_theta(driving_style)
        super().__init__(episode, scenario_map, ego)

    def get_trajectory_one_timestep(self, time):
        """generate the trajectory for one agent"""
        self.reset(reset_time=time)

        buffer_scene = self.get_buffer_scene(time, save_planned_tra=True)

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

        # the first value of idx indicates the highest probability
        if len(idx) < 1:
            return None

        tra = buffer_scene[idx[0]]

        return tra[0]


class CreateAnimation:
    def __init__(self, scenario_map):
        self.fig, self.ax = plt.subplots()
        plot_map(scenario_map, ax=self.ax, markings=True, midline=False, drivable=True,
                 plot_background=False)
        self.vehicles = {}
        self._num = 0

    def update_and_show(self, agents, time, delta_t):
        # Update each vehicle's trajectory based on the current states
        for vehicle_id, agent_reset_state in agents.items():
            tra = agent_reset_state.trj

            if vehicle_id not in self.vehicles:
                # If this vehicle hasn't been plotted yet, create a Line2D object for it
                self.vehicles[vehicle_id] = Line2D([], [], linestyle='-', marker='o')
                self.ax.add_line(self.vehicles[vehicle_id])

            # plot current position
            inx = min(round((time - agent_reset_state.state.time) / delta_t), len(tra) - 1)
            # update position
            agent_reset_state.state.position = tra[inx][0]
            agent_reset_state.state.velocity = tra[inx][1]
            agent_reset_state.state.acceleration = tra[inx][2]
            agent_reset_state.state.heading = tra[inx][3]

            self.vehicles[vehicle_id].set_data([tra[inx][0][0]], [tra[inx][0][1]])

        # remove inactive vehicles
        keys_to_remove = [aid for aid in self.vehicles.keys() if aid not in agents.keys()]
        for aid in keys_to_remove:
            vehicle_line = self.vehicles.pop(aid)
            vehicle_line.remove()  # This removes the Line2D object from the axes

        self.ax.set_title(f'T = {time}')
        plt.draw()
        plt.pause(0.01)
        self.save_frames()
        self._num += 1

    def save_frames(self):
        """Save the frames for creating git"""
        temp_dir = 'temp_frames'
        os.makedirs(temp_dir, exist_ok=True)
        frame_filename = f'{temp_dir}/frame_{self._num}.png'
        self.fig.savefig(frame_filename)


def main():
    args = parse_args()
    driving_styles = {0: 'Cautious', 1: 'Normal', 2: 'Aggressive', -1: 'All'}
    driving_style = driving_styles[args.driving_style_idx]
    logger.info(f'Loading {driving_style} for visualization.')

    # we do the evaluation here
    test_data = load_dataset(get_config_path(args.map), ['test'])

    for episode in test_data.scenario.episodes:
        scenario_map = Map.parse_from_opendrive(episode.map_file)
        animation = CreateAnimation(scenario_map)

        active_agents = {}
        # define the trajectory re_plan frequency, counted by num * delta_t
        re_plan_frequency = 100 * IRLEnv.delta_t
        re_plan_frequency = min(re_plan_frequency, IRLEnv.forward_simulation_time)
        for frame in episode.frames:
            # for each agent generate trajectory using the weights from IRL.
            for aid in frame.agents.keys():
                re_plan = False
                if aid in active_agents.keys():
                    if frame.time - active_agents[aid].state.time <= re_plan_frequency:
                        continue
                    else:
                        re_plan = True

                irl_eva = IRLEva(episode=episode, scenario_map=scenario_map,
                                 ego=episode.agents[aid], driving_style=driving_style)
                # re_plan but use previous state instead of state from the dataset
                if re_plan:
                    irl_eva.reset_ego_state = active_agents[aid].state

                trj = irl_eva.get_trajectory_one_timestep(frame.time)
                if trj is None:
                    continue

                state = AgentState(time=frame.time, position=trj[0][0], velocity=trj[0][1],
                                   acceleration=trj[0][2], heading=trj[0][3])
                agent_reset_state = AgentReset(state, trj)
                active_agents[aid] = agent_reset_state

            # remove the agents that have reached the end
            keys_to_remove = [aid for aid in active_agents.keys() if aid not in frame.agents.keys()]
            for aid in keys_to_remove:
                active_agents.pop(aid)

            animation.update_and_show(active_agents, frame.time, IRLEnv.delta_t)
            pass


if __name__ == "__main__":
    main()

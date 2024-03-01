import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from typing import Tuple

from sim4ad.irlenv.vehicle.humandriving import HumanLikeVehicle, DatasetVehicle
from sim4ad.opendrive import plot_map


class IRLEnv:
    forward_simulation_time = 5  # time_horizon
    delta_t = 0.033366700033366704

    def __init__(self, episode, scenario_map, ego, IDM):
        self.human = False
        self.done = False
        self.steps = None
        self.episode = episode
        self.scenario_map = scenario_map
        self.IDM = IDM
        self.ego = ego
        self.duration = None
        self.time = 0
        self.vehicles = []
        self.other_agents = {}
        self.interval = []
        self.active_vehicles = []

    def reset(self, reset_time: float, human=False):
        """
        Reset the environment at a given time (scene) and specify whether to use human target
        """
        self.vehicles.clear()
        self.other_agents.clear()
        self.interval.clear()
        self.human = human
        self._create_vehicles(reset_time)
        self.steps = 0
        self.time = 0

    @staticmethod
    def process_raw_trajectory(agent):
        """put x, y and speed in one array"""
        trajectory = []
        for inx, t in enumerate(agent.time):
            trajectory.append(
                [agent.x_vec[inx], agent.y_vec[inx], agent.vx_vec[inx], agent.vy_vec[inx], agent.psi_vec[inx]])

        return np.array(trajectory)

    def _create_vehicles(self, reset_time: float):
        """
        Create ego vehicle and dataset vehicles.
        """
        reset_inx = self.ego.next_index_of_specific_time(reset_time)
        whole_trajectory = self.process_raw_trajectory(self.ego)
        ego_trajectory = whole_trajectory[reset_inx:]
        ego_acc = np.array([self.ego.ax_vec[reset_inx], self.ego.ay_vec[reset_inx]])
        heading = self.ego.psi_vec[reset_inx]
        # get position, velocity and acceleration at the reset time
        self.vehicle = HumanLikeVehicle.create(self.scenario_map, self.ego.UUID, ego_trajectory[0][:2], self.ego.length,
                                               self.ego.width,
                                               ego_trajectory, heading=heading, acceleration=ego_acc,
                                               velocity=ego_trajectory[0][2:4],
                                               human=self.human, IDM=self.IDM)
        self.vehicles.append(self.vehicle)

        # identify the start and end frame in frames
        start_frame = end_frame = None
        for inx in range(len(self.episode.frames) - 1):
            if self.episode.frames[inx].time == reset_time:
                start_frame = inx
            if start_frame is not None:
                end_frame = inx
                if self.episode.frames[inx].time >= self.ego.time[-1] or \
                        self.episode.frames[inx].time - reset_time >= self.forward_simulation_time:
                    break

        self.interval = [start_frame, end_frame]

        # create a set for other vehicles which are existing during the time horizon
        logger.error("No living length is found for ego!") if start_frame is None or end_frame is None else None
        for frame in self.episode.frames[start_frame:end_frame + 1]:

            # select the other agents which appear at the same time as the ego and have the same driving direction
            for aid, agent in frame.agents.items():
                agent_lane = self.scenario_map.best_lane_at(point=agent.position, heading=agent.heading)
                if aid != self.vehicle.vehicle_id and agent_lane.id * self.vehicle.lane.id > 0:
                    if aid not in self.other_agents:
                        self.other_agents[aid] = []
                    self.other_agents[aid].append(agent)

        # create dataset vehicles
        for aid, agent in self.other_agents.items():
            heading = agent[0].heading
            length = agent[0].metadata.length
            width = agent[0].metadata.width
            other_trajectory = np.array(
                [np.concatenate((state.position, state.velocity, state.heading), axis=None) for state in agent])
            dataset_vehicle = DatasetVehicle.create(self.scenario_map, aid, agent[0].position,
                                                    length, width, other_trajectory, heading=heading,
                                                    velocity=agent[0].velocity)
            self.vehicles.append(dataset_vehicle)

    def step(self, action=None):
        """
        Perform an MDP step
        """

        features = self._simulate(action)
        terminal = self._is_terminal()

        info = {
            "velocity": self.vehicle.velocity,
            "crashed": self.vehicle.crashed,
            'offroad': not self.vehicle.on_road,
            "action": action,
            "time": self.time
        }

        return features, terminal, info

    def _simulate(self, action, debug=True) -> np.ndarray:
        """
        Perform several steps of simulation with the planned trajectory
        """
        trajectory_features = []
        # adjust the forward simulation time according the distance to road end
        time_horizon = self.forward_simulation_time

        # generate simulated trajectory
        if action is not None:  # sampled goal
            self.vehicle.trajectory_planner(target_point=action[0], target_speed=action[1],
                                            time_horizon=time_horizon, delta_t=self.ego.delta_t)
        else:  # human goal
            # TODO: change to the human goal
            self.vehicle.trajectory_planner(
                self.vehicle.dataset_traj[self.vehicle.sim_steps + time_horizon / self.delta_t][1],
                (self.vehicle.dataset_traj[self.vehicle.sim_steps + time_horizon / self.delta_t][0] -
                 self.vehicle.dataset_traj[self.vehicle.sim_steps + time_horizon / self.delta_t - 1][
                     0]) / self.delta_t, time_horizon)

        self.run_step = 1

        # forward simulation
        features = None
        for frame_inx in range(self.interval[0], self.interval[1] + 1):
            self.active_vehicles.clear()
            self.act(step=self.run_step, frame_inx=frame_inx)
            self.step_forward(self.delta_t)

            self.time += 1
            self.run_step += 1
            features = self._features()
            trajectory_features.append(features)

            # show the forward simulation
            if debug and self.time % 5 == 0:
                plot_map(self.scenario_map, markings=True, midline=False, drivable=True, plot_background=False)
                plt.plot(self.vehicle.planned_trajectory[:, 0], self.vehicle.planned_trajectory[:, 1], 'b',
                         linewidth=1)
                for vehicle in self.active_vehicles:
                    ego_traj = vehicle.traj.reshape(-1, 2)
                    if isinstance(vehicle, HumanLikeVehicle):
                        plt.scatter(ego_traj[:, 0], ego_traj[:, 1], color='#ADD8E6', s=10)
                    else:
                        plt.plot(vehicle.dataset_traj[:, 0], vehicle.dataset_traj[:, 1], color="y", linewidth=1)
                        plt.scatter(ego_traj[:, 0], ego_traj[:, 1], color='orange', s=10)
                        if vehicle.overtaken:
                            plt.scatter(ego_traj[vehicle.overtaken_inx:, 0], ego_traj[vehicle.overtaken_inx:, 1],
                                        color='r', s=10)
                    plt.xlabel('x')
                    plt.ylabel('y')
                    plt.title(f't={self.episode.frames[frame_inx].time}')
                    # plt.savefig(f"frame_{self.run_step}.png")  # Save each frame as an image
                plt.show()

            # Stop at terminal states
            if self._is_terminal():
                break

        human_likeness = features[-1]
        trajectory_features = np.sum(trajectory_features, axis=0)
        trajectory_features[-1] = human_likeness

        return trajectory_features

    def act(self, step: int, frame_inx: int):
        """
        Decide the actions of each entity on the road.
        """
        self.active_vehicles.append(self.vehicles[0])
        # determine the active non-ego surrounding agents
        simulation_time = self.episode.frames[frame_inx].time
        for aid, agent in self.other_agents.items():
            for vehicle in self.vehicles:
                if aid == vehicle.vehicle_id and agent[0].time <= simulation_time < agent[-1].time:
                    self.active_vehicles.append(vehicle)
        # TODO: active_vehicles are changed during the for-loop
        for vehicle in self.active_vehicles:
            if isinstance(vehicle, DatasetVehicle):
                vehicle.act(self.active_vehicles)
            else:
                vehicle.act(step, self.delta_t)

    def step_forward(self, dt: float):
        """
        Step the dynamics of each entity on the road.

        :param dt: timestep [s]
        """
        for vehicle in self.active_vehicles:
            if isinstance(vehicle, DatasetVehicle):
                vehicle.step(dt, self.active_vehicles)
            else:
                vehicle.step(dt)

        for vehicle in self.active_vehicles:
            for other in self.active_vehicles:
                vehicle.check_collision(other)

    def _is_terminal(self):
        """
        The episode is over if the ego vehicle crashed or go off road or the time is out.
        """
        self.duration = self.interval[1] - self.interval[0]
        return self.vehicle.crashed or self.run_step >= self.duration or not self.vehicle.on_road

    def sampling_space(self) -> Tuple[np.ndarray, np.ndarray]:
        """
            The target sampling space (longitudinal speed and lateral offset)
        """
        current_speed = np.sqrt(self.vehicle.velocity[0] ** 2 + self.vehicle.velocity[1] ** 2)

        # remove lateral offsets that are out of the main lanes
        lane_section = self.vehicle.lane.lane_section
        land_width = self.vehicle.lane.widths[0].constant_width
        if self.vehicle.lane == lane_section.right_lanes[1] or self.vehicle.lane == lane_section.left_lanes[1]:
            lateral_offsets = np.array([0 - land_width, 0])
        elif self.vehicle.lane == lane_section.right_lanes[-2] or self.vehicle.lane == lane_section.left_lanes[-2]:
            lateral_offsets = np.array([0, 0 - land_width])
        else:
            lateral_offsets = np.array([0 - land_width, 0, 0 + land_width])

        min_speed = current_speed - 5 if current_speed > 5 else 0
        max_speed = current_speed + 5
        target_speeds = np.linspace(min_speed, max_speed, 5)

        return lateral_offsets, target_speeds

    def _get_thw(self) -> Tuple[float, float]:
        """Determine the thw for front and rear vehicle"""
        front_vehicle, rear_vehicle = DatasetVehicle.get_front_rear_vehicle(self.active_vehicles, self.vehicle)
        thw_front = front_vehicle[1] / self.vehicle.velocity[0]
        thw_rear = -rear_vehicle[1] / rear_vehicle[0].velocity[0] if rear_vehicle[0] is not None else np.inf
        thw_front = np.exp(-1 / thw_front)
        thw_rear = np.exp(-1 / thw_rear)

        return thw_front, thw_rear

    def _features(self) -> np.ndarray:
        """
        Hand-crafted features
        :return: the array of the defined features
        """
        # ego motion
        ego_longitudial_speeds = np.array(self.vehicle.velocity_history)[:, 0] if self.time >= 3 else [0]
        ego_longitudial_accs = (ego_longitudial_speeds[1:] - ego_longitudial_speeds[
                                                             :-1]) / self.delta_t if self.time >= 3 else [
            0]
        ego_longitudial_jerks = (ego_longitudial_accs[1:] - ego_longitudial_accs[
                                                            :-1]) / self.delta_t if self.time >= 3 else [0]

        ego_lateral_speeds = np.array(self.vehicle.velocity_history)[:, 1] if self.time >= 3 else [0]
        ego_lateral_accs = (ego_lateral_speeds[1:] - ego_lateral_speeds[:-1]) / self.delta_t if self.time >= 3 else [0]

        # travel efficiency
        ego_speed = abs(ego_longitudial_speeds[-1])

        # comfort
        ego_longitudial_acc = ego_longitudial_accs[-1]
        ego_lateral_acc = ego_lateral_accs[-1]
        ego_longitudial_jerk = ego_longitudial_jerks[-1]

        # time headway front (thws_front) and time headway behind (thws_rears)
        thw_front, thw_rear = self._get_thw()

        # avoid collision
        collision = 1 if self.vehicle.crashed or not self.vehicle.on_road else 0

        # interaction (social) impact
        social_impact = 0
        for v in self.active_vehicles:
            if isinstance(v, DatasetVehicle) and v.overtaken and (v.velocity[0] != 0 or v.velocity[1] != 0):
                social_impact += np.abs(v.velocity[0] - v.velocity_history[-1][0]) / self.delta_t if v.velocity[0] - \
                                                                                                     v.velocity_history[
                                                                                                         -1][
                                                                                                         0] < 0 else 0

        # ego vehicle human-likeness
        ego_likeness = self.vehicle.calculate_human_likeness()

        # feature array
        features = np.array([ego_speed, abs(ego_longitudial_acc), abs(ego_lateral_acc), abs(ego_longitudial_jerk),
                             thw_front, thw_rear, collision, social_impact, ego_likeness])

        return features

    # @property
    # def position(self) -> np.ndarray:
    #     """ Get all LaneBorders of this Lane """
    #     return self._position

import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from typing import Tuple

from sim4ad.irlenv.vehicle.humandriving import HumanLikeVehicle, DatasetVehicle
from sim4ad.opendrive import plot_map
from sim4ad.irlenv import utils


class IRLEnv:
    forward_simulation_time = 5  # time_horizon
    delta_t = 0.033366700033366704

    def __init__(self, episode, scenario_map, ego=None, idm=False):
        self.human = False
        self.episode = episode
        self.scenario_map = scenario_map
        self.IDM = idm
        self.ego = ego
        self.time = 0
        self.run_step = 0
        self.vehicles = []
        self.active_vehicles = []
        self.reset_time = None
        self.start_frame = None

    def reset(self, reset_time, human=False):
        """
        Reset the environment at a given time (scene) and specify whether to use human target
        """
        self.vehicles.clear()
        self.human = human
        self._create_vehicles(reset_time)
        self.run_step = 0
        self.time = 0
        self.reset_time = reset_time

    @staticmethod
    def process_raw_trajectory(agent):
        """put x, y and speed in one array"""
        trajectory = []
        for inx, t in enumerate(agent.time):
            trajectory.append(
                [agent.x_vec[inx], agent.y_vec[inx], agent.vx_vec[inx], agent.vy_vec[inx], agent.psi_vec[inx]])

        return np.array(trajectory)

    def _create_dataset_vehicle(self, agent):
        aid, state = agent
        agent_lane = self.scenario_map.best_lane_at(point=state.position, heading=state.heading)
        dataset_vehicle = None
        if aid != self.vehicle.vehicle_id and agent_lane is not None and agent_lane.id * self.vehicle.lane.id > 0:
            heading = state.heading
            length = state.metadata.length
            width = state.metadata.width
            agent = self.episode.agents[aid]
            reset_inx = agent.next_index_of_specific_time(self.reset_time)
            whole_trajectory = self.process_raw_trajectory(agent)
            other_trajectory = whole_trajectory[reset_inx:]
            dataset_vehicle = DatasetVehicle.create(self.scenario_map, aid, agent[0].position,
                                                    length, width, other_trajectory, heading=heading,
                                                    velocity=agent[0].velocity)

        return dataset_vehicle

    def _create_vehicles(self, reset_time):
        """
        Create ego vehicle and dataset vehicles.
        """
        # identify the start and end frame in frames
        for inx in range(len(self.episode.frames) - 1):
            if self.episode.frames[inx].time == reset_time:
                self.start_frame = inx
                break

        # use human trajectory to compute ego likeness
        reset_inx = self.ego.next_index_of_specific_time(reset_time)
        whole_trajectory = self.process_raw_trajectory(self.ego)
        ego_trajectory = whole_trajectory[reset_inx:]

        # get position, velocity and acceleration at the reset time
        ego_state = self.episode.frames[self.start_frame].agents[self.ego.UUID]
        ego_acc = ego_state.acceleration
        heading = ego_state.heading
        self.vehicle = HumanLikeVehicle.create(self.scenario_map, self.ego.UUID, ego_state.position, self.ego.length,
                                               self.ego.width,
                                               ego_trajectory, heading=heading, acceleration=ego_acc,
                                               velocity=ego_state.velocity,
                                               human=self.human, IDM=self.IDM)
        self.vehicles.append(self.vehicle)

        # create a set for other vehicles which are existing during the time horizon
        logger.error("No living length is found for ego!") if self.start_frame is None else None
        frame = self.episode.frames[self.start_frame]

        # select the other agents which appear at the same time as the ego and have the same driving direction
        for aid, state in frame.agents.items():
            dataset_vehicle = self._create_dataset_vehicle((aid, state))
            if dataset_vehicle is not None:
                self.vehicles.append(dataset_vehicle)

    def step(self, action=None, debug=False):
        """
        Perform an MDP step
        """
        features = self._simulate(action, debug)
        terminal = self._is_terminal()

        info = {
            "velocity": self.vehicle.velocity,
            "crashed": self.vehicle.crashed,
            'offroad': not self.vehicle.on_road,
            "action": action,
            "time": self.time
        }

        return features, terminal, info

    def _get_target_state(self):
        """get the target state for human trajectories"""
        for inx in range(len(self.episode.frames) - 1):
            if self.episode.frames[inx].time >= self.ego.time[-1] or \
                    self.episode.frames[inx].time - self.reset_time >= self.forward_simulation_time:
                return self.episode.frames[inx].agents[self.ego.UUID]

    def _simulate(self, action, debug) -> np.ndarray:
        """
        Perform several steps of simulation with the planned trajectory
        """
        trajectory_features = []
        # adjust the forward simulation time according the distance to road end
        time_horizon = self.forward_simulation_time

        # generate simulated trajectory
        if action is not None:  # sampled goal
            self.vehicle.trajectory_planner(target_point=action[0], target_speed=action[1],
                                            time_horizon=time_horizon, delta_t=self.delta_t)
        # generate human trajectory given initial and final state from the dataset
        else:
            ego_agent = self._get_target_state()
            target_point = utils.local2frenet(point=ego_agent.position, reference_line=self.vehicle.lane.midline)
            self.vehicle.trajectory_planner(target_point=target_point[1], target_speed=ego_agent.speed,
                                            time_horizon=time_horizon, delta_t=self.delta_t)

        self.run_step = 1

        # the first point of simulated trajectory should be close to the planned trajectory
        if len(self.vehicle.traj) == 1:
            dis = np.subtract(self.vehicle.traj[0], self.vehicle.planned_trajectory[0])
            dis = np.sqrt(dis[0] ** 2 + dis[1] ** 2)
            assert dis < 0.2, "Simulated trajectory does not match the planned trajectory."

        # forward simulation
        features = None
        while not self._is_terminal() and self.run_step < len(self.vehicle.planned_trajectory):
            self.active_vehicles.clear()
            self.act(step=self.run_step)
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
                    ego_traj = np.array(vehicle.traj)
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
                    plt.title(f't={self.episode.frames[self.start_frame + self.run_step].time}')
                    # plt.savefig(f"frame_{self.run_step}.png")  # Save each frame as an image
                plt.show()

        human_likeness = features[-1]
        trajectory_features = np.sum(trajectory_features, axis=0)
        trajectory_features[-1] = human_likeness

        return trajectory_features

    def act(self, step: int):
        """
        Decide the actions of each entity on the road.
        """
        # determine active vehicle in current scene
        for aid, agent in self.episode.frames[self.start_frame + self.run_step].agents:
            new_agent = True
            for inx, vehicle in enumerate(self.vehicles):
                if aid == vehicle.vehicle_id:
                    self.active_vehicles.append(self.vehicles[inx])
                    new_agent = False
            # create a new dataset vehicle
            if new_agent:
                dataset_vehicle = self._create_dataset_vehicle((aid, agent))
                if dataset_vehicle is not None:
                    self.active_vehicles.append(dataset_vehicle)

        # determine act for each vehicle
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
        The episode is over if the ego vehicle crashed or go off the road or the time is out.
        """
        return self.vehicle.crashed or not self.vehicle.on_road

    def sampling_space(self) -> Tuple[np.ndarray, np.ndarray]:
        """
            The target sampling space (longitudinal speed and lateral offset)
        """
        current_speed = np.sqrt(self.vehicle.velocity[0] ** 2 + self.vehicle.velocity[1] ** 2)

        # remove lateral offsets that are out of the main lanes
        lane_section = self.vehicle.lane.lane_section
        lane_width = self.vehicle.lane.widths[0].constant_width
        if self.vehicle.lane == lane_section.left_lanes[-2]:
            lateral_offsets = np.array([0, 0 + lane_width])
        elif self.vehicle.lane == lane_section.left_lanes[1]:
            lateral_offsets = np.array([0 - lane_width, 0])
        elif self.vehicle.lane == lane_section.right_lanes[-2]:
            lateral_offsets = np.array([0, 0 + lane_width])
        elif self.vehicle.lane == lane_section.right_lanes[1]:
            lateral_offsets = np.array([0 - lane_width, 0])
        else:
            lateral_offsets = np.array([0 - lane_width, 0, 0 + lane_width])

        min_speed = current_speed - 5 if current_speed > 5 else 0
        max_speed = current_speed + 5
        target_speeds = np.linspace(min_speed, max_speed, 5)

        return lateral_offsets, target_speeds

    def _get_thw(self) -> Tuple[float, float]:
        """Determine the thw for front and rear vehicle"""
        front_vehicle, rear_vehicle = DatasetVehicle.get_front_rear_vehicle(self.active_vehicles, self.vehicle)
        thw_front = front_vehicle[1] / self.vehicle.velocity[0] if front_vehicle[0] is not None else np.inf
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
        ego_long_speeds = np.array(self.vehicle.velocity_history)[:, 0] if self.time >= 3 else [0]
        ego_long_accs = (ego_long_speeds[1:] - ego_long_speeds[:-1]) / self.delta_t if self.time >= 3 else [0]
        ego_long_jerks = (ego_long_accs[1:] - ego_long_accs[:-1]) / self.delta_t if self.time >= 3 else [0]

        ego_lateral_speeds = np.array(self.vehicle.velocity_history)[:, 1] if self.time >= 3 else [0]
        ego_lateral_accs = (ego_lateral_speeds[1:] - ego_lateral_speeds[:-1]) / self.delta_t if self.time >= 3 else [0]

        # travel efficiency
        ego_speed = abs(ego_long_speeds[-1])

        # comfort
        ego_long_acc = ego_long_accs[-1]
        ego_lat_acc = ego_lateral_accs[-1]
        ego_long_jerk = ego_long_jerks[-1]

        # time headway front (thw_front) and time headway behind (thw_rear)
        thw_front, thw_rear = self._get_thw()

        # avoid collision
        collision = 1 if self.vehicle.crashed or not self.vehicle.on_road else 0

        # interaction (social) impact
        social_impact = 0
        for v in self.active_vehicles:
            if isinstance(v, DatasetVehicle) and v.overtaken and (v.velocity[0] != 0 or v.velocity[1] != 0):
                social_impact += np.abs(v.velocity[0] - v.velocity_history[-1][0]) / self.delta_t \
                    if v.velocity[0] - v.velocity_history[-1][0] < 0 else 0

        # ego vehicle human-likeness
        ego_likeness = self.vehicle.calculate_human_likeness()

        # feature array
        features = np.array([ego_speed, abs(ego_long_acc), abs(ego_lat_acc), abs(ego_long_jerk),
                             thw_front, thw_rear, collision, social_impact, ego_likeness])

        return features

    def get_buffer_scene(self, t, save_planned_tra=False):
        """Get the features of sampled trajectories"""
        # set up buffer of the scene
        buffer_scene = []

        lateral_offsets, target_speeds = self.sampling_space()
        # for each lateral offset and target_speed combination
        for lateral in lateral_offsets:
            for target_speed in target_speeds:
                action = (lateral, target_speed)
                features, terminated, info = self.step(action)

                # get the features
                traj_features = features[:-1]
                human_likeness = features[-1]

                # add scene trajectories to buffer
                if save_planned_tra:
                    buffer_scene.append((self.vehicle.traj, None, traj_features, human_likeness))
                else:
                    buffer_scene.append((lateral, target_speed, traj_features, human_likeness))

                # set back to previous step
                self.reset(reset_time=t)

        return buffer_scene

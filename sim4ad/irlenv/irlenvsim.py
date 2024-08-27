import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from typing import Tuple, Optional
import joblib
from sim4ad.irlenv.vehicle.humandriving import HumanLikeVehicle, DatasetVehicle
from sim4ad.opendrive import plot_map
from sim4ad.irlenv import utils
from simulator.state_action import State
from simulator.simulator_util import compute_distance_markings


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
        self.active_vehicles = []
        self.reset_time = None
        self.start_frame = None
        self.reset_ego_state = None
        # using exponential normalization, no longer necessary to load mean and std
        # self._feature_mean_std = self.load_feature_normalization()

    def reset(self, reset_time, human=False):
        """
        Reset the environment at a given time (scene) and specify whether to use human target
        """
        self.reset_time = reset_time
        self.active_vehicles.clear()
        self.human = human
        self._create_vehicles(reset_time)
        self.run_step = 0
        self.time = 0

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
        if agent_lane is not None and agent_lane.id * self.vehicle.lane.id > 0:
            heading = state.heading
            length = state.metadata.length
            width = state.metadata.width

            # get the dataset trajectory from the reset_index
            agent = self.episode.agents[aid]
            reset_inx = agent.next_index_of_specific_time(self.reset_time)
            whole_trajectory = self.process_raw_trajectory(agent)
            other_trajectory = whole_trajectory[reset_inx:]

            dataset_vehicle = DatasetVehicle.create(self.scenario_map, aid, state.position,
                                                    length, width, other_trajectory, heading=heading,
                                                    velocity=state.velocity)

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
        # evaluation: state from previous state; training: state from dataset
        if self.reset_ego_state is not None:
            ego_state = self.reset_ego_state
        else:
            ego_state = self.episode.frames[self.start_frame].agents[self.ego.UUID]
        ego_acc = ego_state.acceleration
        heading = ego_state.heading
        self.vehicle = HumanLikeVehicle.create(self.scenario_map, self.ego.UUID, ego_state.position, self.ego.length,
                                               self.ego.width,
                                               ego_trajectory, heading=heading, acceleration=ego_acc,
                                               velocity=ego_state.velocity,
                                               human=self.human, idm=self.IDM)
        self.active_vehicles.append(self.vehicle)

        # create a set for other vehicles which are existing during the time horizon
        logger.error("No living length is found for ego!") if self.start_frame is None else None
        frame = self.episode.frames[self.start_frame]

        # select the other agents which appear at the same time as the ego and have the same driving direction
        for aid, state in frame.agents.items():
            if aid != self.vehicle.vehicle_id:
                dataset_vehicle = self._create_dataset_vehicle((aid, state))
                if dataset_vehicle is not None:
                    self.active_vehicles.append(dataset_vehicle)

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
        for inx in range(len(self.episode.frames)):
            # smaller than the delta_t, it means they are very close
            if self.ego.time[-1] - self.episode.frames[inx].time < 0.03:
                return self.episode.frames[inx].agents[self.ego.UUID], self.ego.time[-1] - self.reset_time
            elif self.episode.frames[inx].time - self.reset_time >= self.forward_simulation_time:
                return self.episode.frames[inx].agents[self.ego.UUID], self.forward_simulation_time

        logger.error(f"aid: {self.ego.UUID}, ego_time: {self.ego.time[-1]}, frame_time: {self.episode.frames[-1].time}")

    def _simulate(self, action, debug) -> Optional[np.ndarray]:
        """
        Perform several steps of simulation with the planned trajectory
        """
        trajectory_features = []

        # generate simulated trajectory
        if action is not None:  # sampled goal
            self.vehicle.trajectory_planner(target_point=action[0], target_speed=action[1],
                                            time_horizon=self.forward_simulation_time, delta_t=self.delta_t)
        # generate human trajectory given initial and final state from the dataset
        else:
            ego_agent, time_horizon = self._get_target_state()
            target_point = utils.local2frenet(point=ego_agent.position, reference_line=self.vehicle.lane.midline)
            self.vehicle.trajectory_planner(target_point=target_point[1], target_speed=ego_agent.speed,
                                            time_horizon=time_horizon, delta_t=self.delta_t)

        self.run_step = 1

        # the first point of simulated trajectory should be close to the planned trajectory
        if len(self.vehicle.traj) == 1:
            dis = np.subtract(self.vehicle.traj[0][0], self.vehicle.planned_trajectory[0])
            dis = np.sqrt(dis[0] ** 2 + dis[1] ** 2)
            assert dis < 0.2, "Simulated trajectory does not match the planned trajectory."

        # forward simulation
        features = None
        total_steps = min(len(self.vehicle.planned_trajectory), len(self.episode.frames) - self.start_frame)
        while not self._is_terminal() and self.run_step < total_steps:
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
                    ego_traj = np.array([trj[0] for trj in vehicle.traj])
                    if isinstance(vehicle, HumanLikeVehicle):
                        plt.scatter(ego_traj[:, 0], ego_traj[:, 1], color='#ADD8E6', s=10)
                    else:
                        plt.plot(vehicle.dataset_traj[:, 0], vehicle.dataset_traj[:, 1], color="y", linewidth=1)
                        plt.scatter(ego_traj[:, 0], ego_traj[:, 1], color='orange', s=10)
                        if vehicle.overridden:
                            plt.scatter(ego_traj[vehicle.overridden_inx:, 0], ego_traj[vehicle.overridden_inx:, 1],
                                        color='r', s=10)
                    plt.xlabel('x')
                    plt.ylabel('y')
                    plt.title(f't={self.episode.frames[self.start_frame + self.run_step].time}')
                    # plt.savefig(f"frame_{self.run_step}.png")  # Save each frame as an image
                plt.show()

        if features is None:
            logger.warning(f'features is None, length of planned tra is {len(self.vehicle.planned_trajectory)}')
            return None

        human_likeness = features[-1]
        trajectory_features = np.sum(trajectory_features, axis=0)
        trajectory_features[-1] = human_likeness

        return trajectory_features

    def act(self, step: int):
        """
        Decide the actions of each entity on the road.
        """
        # determine active vehicle in current scene
        previous_active_vehicle_ids = [vehicle.vehicle_id for vehicle in self.active_vehicles]
        current_active_vehicle_ids = []
        for aid, agent in self.episode.frames[self.start_frame + self.run_step].agents.items():
            # create a new dataset vehicle
            if aid not in previous_active_vehicle_ids:
                dataset_vehicle = self._create_dataset_vehicle((aid, agent))
                if dataset_vehicle is not None:
                    self.active_vehicles.append(dataset_vehicle)

            # remove the previous active vehicle which has reached the last step
            else:
                index = previous_active_vehicle_ids.index(aid)
                if self.active_vehicles[index].sim_steps >= len(self.active_vehicles[index].dataset_traj) - 1:
                    continue
            current_active_vehicle_ids.append(aid)

        # delete inactive vehicles
        self.active_vehicles = [vehicle for vehicle in self.active_vehicles if
                                vehicle.vehicle_id in current_active_vehicle_ids]

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
        ego_long_acc = abs(ego_long_accs[-1])
        ego_lat_acc = abs(ego_lateral_accs[-1])
        ego_long_jerk = abs(ego_long_jerks[-1])

        # time headway front (thw_front) and time headway behind (thw_rear)
        thw_front, thw_rear = self._get_thw()

        # centerline deviation
        # _, d = utils.local2frenet(self.vehicle.position, self.vehicle.lane.midline)
        # d_centerline = abs(d)

        # lateral distance to the nearest lane marker
        state = State(time=self.vehicle.timer, position=self.vehicle.position, velocity=self.vehicle.velocity[0],
                      acceleration=self.vehicle.acceleration[0], heading=self.vehicle.heading,
                      lane=self.vehicle.lane, agent_width=self.vehicle.WIDTH, agent_length=self.vehicle.LENGTH)
        distance_left_lane_marking, distance_right_lane_marking = compute_distance_markings(state=state)
        nearest_distance_lane_marking = min(abs(distance_left_lane_marking), abs(distance_right_lane_marking))

        # ego vehicle human-likeness
        ego_likeness = self.vehicle.calculate_human_likeness()

        # feature array
        features = np.array([ego_speed, ego_long_acc, ego_lat_acc, ego_long_jerk,
                             thw_front, thw_rear, nearest_distance_lane_marking])

        # normalize features using exponential
        normalized_features = self.exponential_normalization(features)
        # add ego likeness for monitoring
        normalized_features = np.append(normalized_features, ego_likeness)
        return normalized_features

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

                if features is None:
                    self.reset(reset_time=t)
                    return buffer_scene

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

    @staticmethod
    def exponential_normalization(features):
        """Using exponential for normalization"""
        normalized_features = [None for _ in range(len(features))]
        for inx, feature in enumerate(features):
            # skip THW
            if inx == 4 or inx == 5:
                normalized_features[inx] = feature
            else:
                normalized_features[inx] = np.exp(-1 / feature) if feature else 0

        return normalized_features

    @staticmethod
    def load_feature_normalization():
        """Loading the mean and standard deviation for feature normalization"""
        return joblib.load('results/feature_normalization.pkl')

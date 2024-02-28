from __future__ import division, print_function

from loguru import logger

import numpy as np
import matplotlib.pyplot as plt
from sim4ad.irlenv.vehicle.control import ControlledVehicle
from sim4ad.irlenv import utils
from sim4ad.irlenv.vehicle.dynamics import Vehicle
from sim4ad.irlenv.vehicle.behavior import IDMVehicle
from sim4ad.irlenv.vehicle.control import MDPVehicle
from sim4ad.irlenv.vehicle.planner import planner


class DatasetVehicle(IDMVehicle):
    """
    Use dataset human driving trajectories.
    """
    # Longitudinal policy parameters
    ACC_MAX = 5.0  # [m/s2]  """Maximum acceleration."""
    COMFORT_ACC_MAX = 3.0  # [m/s2]  """Desired maximum acceleration."""
    COMFORT_ACC_MIN = -3.0  # [m/s2] """Desired maximum deceleration."""
    DISTANCE_WANTED = 1.0  # [m] """Desired jam distance to the front vehicle."""
    TIME_WANTED = 0.5  # [s]  """Desired time gap to the front vehicle."""
    DELTA = 4.0  # [] """Exponent of the velocity term."""

    # Lateral policy parameters [MOBIL]
    POLITENESS = 0.1  # in [0, 1]
    LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0  # [m/s2]
    LANE_CHANGE_DELAY = 1.0  # [s]

    def __init__(self, scenario_map, position,
                 heading,
                 velocity,
                 target_lane=None,
                 target_velocity=None,
                 route=None,
                 enable_lane_change=False,  # only changed here
                 timer=None,
                 vehicle_ID=None, v_length=None, v_width=None, dataset_traj=None):
        super(DatasetVehicle, self).__init__(scenario_map, position, heading, velocity, target_lane, target_velocity,
                                             route, enable_lane_change, timer)

        self.dataset_traj = dataset_traj
        self.traj = np.array(self.position)
        self.vehicle_ID = vehicle_ID
        self.sim_steps = 0
        self.overtaken = False
        self.appear = True if self.position[0] != 0 else False
        self.velocity_history = []
        self.heading_history = []
        self.crash_history = []
        self.overtaken_history = []

        # Vehicle length [m]
        self.LENGTH = v_length
        # Vehicle width [m]
        self.WIDTH = v_width

    @classmethod
    def create(cls, scenario_map, vehicle_ID, position, v_length, v_width, dataset_traj, heading, velocity):
        """
        Create a new dataset vehicle .

        :param scenario_map: the road where the vehicle is driving
        :param vehicle_ID: dataset vehicle ID
        :param position: the position where the vehicle start on the road
        :param v_length: vehicle length
        :param v_width: vehicle width
        :param dataset_traj: dataset trajectory
        :param velocity: initial velocity in [m/s]. If None, will be chosen randomly
        :param heading: initial heading

        :return: A vehicle with dataset position and velocity
        """

        v = cls(scenario_map, position, heading, velocity, vehicle_ID=vehicle_ID, v_length=v_length, v_width=v_width,
                dataset_traj=dataset_traj)

        return v

    @staticmethod
    def get_front_rear_vehicle(vehicles, subject):
        """Get the front and rear vehicle of ego"""
        dis_front = np.inf
        dis_rear = -np.inf
        front_vehicle = rear_vehicle = None
        for vehicle in vehicles:
            if vehicle.vehicle_ID == subject.vehicle_ID:
                continue
            if vehicle.lane == subject.lane:
                # check the s value along the lane
                s, d = utils.local2frenet(vehicle.position, subject.lane.midline)
                if s > subject.s and s - subject.s < dis_front:
                    dis_front = s - subject.s
                    front_vehicle = vehicle
                elif s < subject.s and s - subject.s > dis_rear:
                    dis_rear = s - subject.s
                    rear_vehicle = vehicle

        return front_vehicle, rear_vehicle, dis_front, dis_rear

    def act(self, active_vehicles):
        """
        Execute an action when the dataset vehicle is overriden.

        :param active_vehicles: the existing vehicles in a scene
        """
        if not self.overtaken or self.crashed:
            return

        action = {}
        front_vehicle, rear_vehicle, _, _, = self.get_front_rear_vehicle(active_vehicles, self)

        # Lateral: MOBIL
        self.follow_road()
        if self.enable_lane_change:
            self.change_lane_policy()
        action['steering'] = self.steering_control(self.target_lane)

        # Longitudinal: IDM
        action['acceleration'] = self.acceleration(ego_vehicle=self, front_vehicle=front_vehicle,
                                                   rear_vehicle=rear_vehicle)
        action['acceleration'] = np.clip(action['acceleration'], -self.ACC_MAX, self.ACC_MAX)
        self.action = action

    def step(self, dt, active_vehicles):
        """
        Update the state of a dataset vehicle.
        If the front vehicle is too close, use IDM model to override the dataset vehicle.
        """
        self.appear = True if self.dataset_traj[self.sim_steps][0] != 0 else False
        self.timer += dt
        self.sim_steps += 1
        self.heading_history.append(self.heading)
        self.velocity_history.append(self.velocity)
        self.crash_history.append(self.crashed)
        self.overtaken_history.append(self.overtaken)

        # Check if need to overtake
        front_vehicle, rear_vehicle, dis_front, _ = self.get_front_rear_vehicle(active_vehicles, self)
        if front_vehicle is not None and isinstance(front_vehicle, DatasetVehicle) and front_vehicle.overtaken:
            gap = dis_front
            desired_gap = self.desired_gap(self, front_vehicle)
        elif front_vehicle is not None and (
                isinstance(front_vehicle, HumanLikeVehicle) or isinstance(front_vehicle, MDPVehicle)):
            gap = dis_front
            desired_gap = self.desired_gap(self, front_vehicle)
        else:
            gap = 100
            desired_gap = 50

        if gap >= desired_gap and not self.overtaken:
            # follow the original dataset trajectory
            self.position = self.dataset_traj[self.sim_steps][:2]
            self.heading = self.dataset_traj[self.sim_steps][-1]
            self.velocity = self.dataset_traj[self.sim_steps][2:4]
            self.target_velocity = self.velocity
            self.lane = self.scenario_map.best_lane_at(point=self.position, heading=self.heading)
            self.s, self.d = utils.local2frenet(point=self.position, reference_line=self.lane.midline)
        else:
            self.overtaken = True
            logger.info(f'Vehicle {self.vehicle_ID} is overtaken!')

            if self.lane.id < 0:
                self.target_lane = self.scenario_map.get_lane(self.lane.parent_road.id, self.lane.id + 1, 0)
            else:
                self.target_lane = self.scenario_map.get_lane(self.lane.parent_road.id, self.lane.id - 1, 0)
            super(DatasetVehicle, self).step(dt)

        self.traj = np.append(self.traj, self.position, axis=0)

    def check_collision(self, other):
        """
        Check for collision with another vehicle.

        :param other: the other vehicle
        """
        if not self.COLLISIONS_ENABLED or not other.COLLISIONS_ENABLED or self.crashed or other is self:
            return

        # Fast spherical pre-check
        if np.linalg.norm(other.position - self.position) > self.LENGTH:
            return

        # if both vehicles are NGSIM vehicles and have not been overriden
        if isinstance(self, DatasetVehicle) and not self.overtaken and isinstance(other,
                                                                                  DatasetVehicle) and not other.overtaken:
            return

            # Accurate rectangular check
        if utils.rotated_rectangles_intersect((self.position, 0.9 * self.LENGTH, 0.9 * self.WIDTH, self.heading),
                                              (other.position, 0.9 * other.LENGTH, 0.9 * other.WIDTH,
                                               other.heading)) and self.appear:
            self.velocity = other.velocity = min([self.velocity, other.velocity], key=abs)
            self.crashed = other.crashed = True


class HumanLikeVehicle(IDMVehicle):
    """
    Create a human-like (IRL) driving agent.
    """
    TAU_A = 0.2  # [s]
    TAU_DS = 0.1  # [s]
    PURSUIT_TAU = 1.5 * TAU_DS  # [s]
    KP_A = 1 / TAU_A
    KP_HEADING = 1 / TAU_DS
    KP_LATERAL = 1 / 0.2  # [1/s]
    MAX_STEERING_ANGLE = np.pi / 3  # [rad]
    MAX_VELOCITY = 50  # [m/s]

    def __init__(self, scenario_map, position,
                 heading,
                 velocity,
                 acceleration,
                 target_lane=None,
                 target_velocity=15,  # Speed reference
                 route=None,
                 timer=None,
                 vehicle_ID=None, v_length=None, v_width=None, dataset_traj=None, human=False, IDM=False):
        super(HumanLikeVehicle, self).__init__(scenario_map, position, heading, velocity, target_lane, target_velocity,
                                               route, timer)

        self.dataset_traj = dataset_traj
        self.traj = np.array(self.position)
        self.sim_steps = 0
        self.vehicle_ID = vehicle_ID
        self.planned_trajectory = None
        self.human = human
        self.IDM = IDM
        self.velocity_history = []
        self.heading_history = []
        self.crash_history = []
        self.acceleration = acceleration
        self.steering_noise = None
        self.acc_noise = None

        self.LENGTH = v_length  # Vehicle length [m]
        self.WIDTH = v_width  # Vehicle width [m]

    @classmethod
    def create(cls, scenario_map, vehicle_ID, position, v_length, v_width, dataset_traj, heading, velocity, acceleration,
               target_velocity=15, human=False, IDM=False):
        """
        Create a human-like (IRL) driving vehicle in replace of a dataset vehicle.
        """
        v = cls(scenario_map, position, heading, velocity, acceleration, target_velocity=target_velocity,
                vehicle_ID=vehicle_ID, v_length=v_length, v_width=v_width, dataset_traj=dataset_traj, human=human, IDM=IDM)

        return v

    def _theta_frenet(self, distance: float):
        """
            Get the rotation angle between vehicle coordinate system and frenet system

            :arg distance: the s value of the agent on frenet
            :return theta_frenet: the angle between the ego and the lane midline
        """
        lane_heading = self.lane.get_heading_at(distance)
        delta_theta = utils.wrap_to_pi(self.heading - lane_heading)

        rotation_matrix = np.array([
            [np.cos(delta_theta), -np.sin(delta_theta)],
            [np.sin(delta_theta), np.cos(delta_theta)]
        ])

        return rotation_matrix

    def trajectory_planner(self, target_point, target_speed, time_horizon, delta_t):
        """
        Plan a trajectory for the human-like (IRL) vehicle.
        """
        rotation_matrix = self._theta_frenet(distance=self.s)

        s_v, d_v = np.dot(rotation_matrix, self.velocity)
        s_a, d_a = np.dot(rotation_matrix, self.acceleration)

        s_d, s_d_d, s_d_d_d = self.s, s_v, s_a  # Longitudinal
        c_d, c_d_d, c_d_dd = self.d, d_v, d_a  # Lateral
        target_area, speed, T = target_point, target_speed, time_horizon

        if not self.human:
            target_area += np.random.normal(0, 0.2)

        path = planner(s_d, s_d_d, s_d_d_d, c_d, c_d_d, c_d_dd, target_area, speed, T, delta_t)

        planned_trajectory_frenet = np.array([[x, y] for x, y in zip(path[0].x, path[0].y)])
        planned_trajectory = []
        for p in planned_trajectory_frenet:
            planned_trajectory.append(utils.frenet2local(reference_line=self.lane.midline, s=p[0], d=p[1]))
        self.planned_trajectory = np.array(planned_trajectory)

        if self.IDM:
            self.planned_trajectory = None

        # if constant velocity:
        # time = np.arange(0, T*10, 1)
        # path_x = self.position[0] + self.velocity * np.cos(self.heading) * time/10
        # path_y = self.position[1] + self.velocity * np.sin(self.heading) * time/10
        # self.planned_trajectory = np.array([[x, y] for x, y in zip(path_x, path_y)])

    def act(self, step, dt):
        if self.planned_trajectory is not None:
            self.action = {'steering': self.steering_control(self.planned_trajectory, step),
                           'acceleration': self.velocity_control(self.planned_trajectory, step, dt)}
        elif self.IDM:
            super(HumanLikeVehicle, self).act()
        else:
            return

    def steering_control(self, trajectory, step):
        """
        Steer the vehicle to follow the given trajectory.

        1. Lateral position is controlled by a proportional controller yielding a lateral velocity command
        2. Lateral velocity command is converted to a heading reference
        3. Heading is controlled by a proportional controller yielding a heading rate command
        4. Heading rate command is converted to a steering angle

        :param trajectory: the trajectory to follow
        :return: a steering wheel angle command [rad]
        """
        target_coords = trajectory[step]
        # transform to frenet coordinate system
        target_coords = utils.local2frenet(point=target_coords, reference_line=self.lane.midline)
        # Lateral position control
        lateral_velocity_command = self.KP_LATERAL * (target_coords[1] - self.d)

        # Lateral velocity to heading
        speed = np.sqrt(self.velocity[0] ** 2 + self.velocity[1] ** 2)
        heading_command = np.arcsin(np.clip(lateral_velocity_command / utils.not_zero(speed), -1, 1))
        heading_ref = np.clip(heading_command, -np.pi / 4, np.pi / 4)

        lane_heading = self.lane.get_heading_at(self.s)
        # Heading control
        heading_rate_command = self.KP_HEADING * utils.wrap_to_pi(heading_ref + lane_heading - self.heading)

        # Heading rate to steering angle
        steering_angle = np.arctan(self.LENGTH / utils.not_zero(speed) * heading_rate_command)
        steering_angle = np.clip(steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)

        return steering_angle

    def velocity_control(self, trajectory, step, dt):
        """
        Control the velocity of the vehicle.

        Using a simple proportional controller.

        :param trajectory: the trajectory to follow
        :return: an acceleration command [m/s2]
        """
        target_velocity = np.array([(trajectory[step][0] - trajectory[step - 1][0]) / dt,
                                   (trajectory[step][1] - trajectory[step - 1][1]) / dt])
        # transform target velocity from local coordinate system to the vehicle coordinate system
        rotation_matrix = np.array([
            [np.cos(self.heading), np.sin(self.heading)],
            [-np.sin(self.heading), np.cos(self.heading)]
        ])
        target_velocity = np.dot(rotation_matrix, target_velocity)

        # acceleration = self.KP_A * (target_velocity - self.velocity)
        acceleration = (target_velocity - self.velocity) / dt

        return acceleration

    def step(self, dt):
        self.sim_steps += 1
        self.heading_history.append(self.heading)
        self.velocity_history.append(self.velocity)
        self.crash_history.append(self.crashed)
        super(HumanLikeVehicle, self).step(dt)

        self.traj = np.append(self.traj, self.position, axis=0)

    def calculate_human_likeness(self):
        original_traj = self.dataset_traj[:self.sim_steps + 1, :2]
        ego_traj = self.traj.reshape(-1, 2)
        ADE = np.mean([np.linalg.norm(original_traj[i] - ego_traj[i]) for i in
                       range(ego_traj.shape[0])])  # Average Displacement Error (ADE)
        FDE = np.linalg.norm(original_traj[-1] - ego_traj[-1])  # Final Displacement Error (FDE)

        return FDE

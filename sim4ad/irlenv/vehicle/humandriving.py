from __future__ import division, print_function

import numpy as np
from typing import Optional

from sim4ad.irlenv import utils
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
                 vehicle_id=None, v_length=None, v_width=None, dataset_traj=None):
        super(DatasetVehicle, self).__init__(scenario_map, position, heading, velocity, target_lane, target_velocity,
                                             route, enable_lane_change, timer, vehicle_id)

        self.dataset_traj = dataset_traj
        self.traj = [[position, velocity, self.acceleration, heading]]
        self.vehicle_id = vehicle_id
        self.sim_steps = 0
        self.overridden = False
        self.appear = True if self.position[0] != 0 else False
        self.velocity_history = []
        self.heading_history = []
        self.crash_history = []
        self.overridden_history = []
        self.overridden_inx = 0

        # Vehicle length [m]
        self.LENGTH = v_length
        # Vehicle width [m]
        self.WIDTH = v_width

    @classmethod
    def create(cls, scenario_map, vehicle_id, position, v_length, v_width, dataset_traj, heading, velocity):
        """
        Create a new dataset vehicle .

        :param scenario_map: the road where the vehicle is driving
        :param vehicle_id: dataset vehicle ID
        :param position: the position where the vehicle start on the road
        :param v_length: vehicle length
        :param v_width: vehicle width
        :param dataset_traj: dataset trajectory
        :param velocity: initial velocity in [m/s]. If None, will be chosen randomly
        :param heading: initial heading

        :return: A vehicle with dataset position and velocity
        """

        v = cls(scenario_map, position, heading, velocity, vehicle_id=vehicle_id, v_length=v_length, v_width=v_width,
                dataset_traj=dataset_traj)

        return v

    def act(self, active_vehicles=None, action=None):
        """
        Execute an action if the dataset vehicle is overriden.

        :param action:
        :param active_vehicles: the existing vehicles in a scene
        """
        if not self.overridden or self.crashed:
            return

        action = {}
        front_vehicle, rear_vehicle = self.get_front_rear_vehicle(active_vehicles, self)

        # Lateral: MOBIL
        if self.enable_lane_change:
            self.change_lane_policy(active_vehicles)
        action['steering'] = self.steering_control(self.target_lane)

        # Longitudinal: IDM
        action['acceleration'] = self.acceleration(ego_vehicle=self, front_vehicle=front_vehicle,
                                                   rear_vehicle=rear_vehicle)
        action['acceleration'] = np.array([np.clip(action['acceleration'], -self.ACC_MAX, self.ACC_MAX), 0])
        self.action = action

    def step(self, dt, active_vehicles=None):
        """
        Update the state of a dataset vehicle.
        If the front vehicle is too close, use IDM model to override the dataset vehicle.
        """
        self.appear = True if self.dataset_traj[self.sim_steps][0] != 0 else False
        self.timer += dt
        self.sim_steps += 1
        self.heading_history.append(self.heading)
        self.velocity_history.append(self.velocity.copy())
        self.crash_history.append(self.crashed)
        self.overridden_history.append(self.overridden)

        # Check if the vehicle is needed to be overridden
        front_vehicle, _ = self.get_front_rear_vehicle(active_vehicles, self)
        if front_vehicle[0] is not None and isinstance(front_vehicle[0], DatasetVehicle) and front_vehicle[0].overridden:
            gap = front_vehicle[1]
            desired_gap = self.desired_gap(self, front_vehicle[0])
        elif front_vehicle[0] is not None and (
                isinstance(front_vehicle[0], HumanLikeVehicle) or isinstance(front_vehicle[0], MDPVehicle)):
            gap = front_vehicle[1]
            desired_gap = self.desired_gap(self, front_vehicle[0])
        else:
            gap = 100
            desired_gap = 50

        if gap >= desired_gap and not self.overridden:
            # follow the original dataset trajectory
            self.position = self.dataset_traj[self.sim_steps][:2].copy()
            self.velocity = self.dataset_traj[self.sim_steps][2:4].copy()
            self.heading = self.dataset_traj[self.sim_steps][-1]
            # target velocity for IDM
            self.target_velocity = self.velocity
            self.lane = self.scenario_map.best_lane_at(point=self.position, heading=self.heading)
            self.s, self.d = utils.local2frenet(point=self.position, reference_line=self.lane.midline)
        else:
            self.overridden = True
            if not self.overridden_history[-1]:
                self.overridden_inx = self.sim_steps

            # target lane for lane changing, if equals to self.lane, will keep current lane
            # self.target_lane = self.lane

            super(DatasetVehicle, self).step(dt)

        self.traj.append([self.position.copy(), self.velocity.copy(), self.acceleration, self.heading])

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

        # if both vehicles are dataset vehicles and have not been overridden
        if isinstance(self, DatasetVehicle) and not self.overridden and \
                isinstance(other, DatasetVehicle) and not other.overridden:
            return

            # Accurate rectangular check
        if utils.rotated_rectangles_intersect((self.position, 0.9 * self.LENGTH, 0.9 * self.WIDTH, self.heading),
                                              (other.position, 0.9 * other.LENGTH, 0.9 * other.WIDTH,
                                               other.heading)) and self.appear:
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
                 target_velocity=None,  # Speed reference
                 route=None,
                 timer=None,
                 vehicle_id=None, v_length=None, v_width=None, dataset_traj=None, human=False, idm=False):
        super(HumanLikeVehicle, self).__init__(scenario_map, position, heading, velocity, target_lane, target_velocity,
                                               route, timer)

        self.dataset_traj = dataset_traj
        self.traj = [[position, velocity, acceleration, heading]]
        self.sim_steps = 0
        self.vehicle_id = vehicle_id
        self.planned_trajectory = None
        self.human = human
        self.IDM = idm
        self.velocity_history = []
        self.heading_history = []
        self.crash_history = []
        self.acceleration = acceleration
        self.steering_noise = None
        self.acc_noise = None

        self.LENGTH = v_length  # Vehicle length [m]
        self.WIDTH = v_width  # Vehicle width [m]

    @classmethod
    def create(cls, scenario_map, vehicle_id, position, v_length, v_width, dataset_traj, heading, velocity,
               acceleration, target_velocity=None, human=False, idm=False):
        """
        Create a human-like (IRL) driving vehicle in replace of a dataset vehicle.
        """
        v = cls(scenario_map, position, heading, velocity, acceleration, target_velocity=target_velocity,
                vehicle_id=vehicle_id, v_length=v_length, v_width=v_width, dataset_traj=dataset_traj, human=human,
                idm=idm)

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
            if p[0] > self.lane.length:
                continue
            point_x, point_y = utils.frenet2local(reference_line=self.lane.midline, s=p[0], d=p[1])
            planned_trajectory.append([point_x, point_y])
        self.planned_trajectory = np.array(planned_trajectory)

        if self.IDM:
            self.planned_trajectory = None

        # if constant velocity:
        # time = np.arange(0, T*10, 1)
        # path_x = self.position[0] + self.velocity * np.cos(self.heading) * time/10
        # path_y = self.position[1] + self.velocity * np.sin(self.heading) * time/10
        # self.planned_trajectory = np.array([[x, y] for x, y in zip(path_x, path_y)])

    def act(self, step: Optional[int] = None, dt: Optional[float] = 0.033366700033366704):
        if self.planned_trajectory is not None:
            self.action = {'steering': self.steering_control(step, dt, self.planned_trajectory),
                           'acceleration': self.velocity_control(step, dt, self.planned_trajectory)}
        elif self.IDM:
            super(HumanLikeVehicle, self).act()
        else:
            return

    def steering_control(self, step: int, dt: Optional[float] = None, trajectory: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Steer the vehicle to follow the given trajectory.

        1. Lateral position is controlled by a proportional controller yielding a lateral velocity command
        2. Lateral velocity command is converted to a heading reference
        3. Heading is controlled by a proportional controller yielding a heading rate command
        4. Heading rate command is converted to a steering angle

        :param step: the index
        :param trajectory: the trajectory to follow
        :param dt
        :return: a steering wheel angle command [rad]
        """
        target_coords = trajectory[step]
        # transform to frenet coordinate system
        target_coords = utils.local2frenet(point=target_coords, reference_line=self.lane.midline)
        # Lateral position control
        # lateral_velocity_command = self.KP_LATERAL * (target_coords[1] - self.d)
        lateral_velocity_command = (target_coords[1] - self.d) / dt

        # Lateral velocity to heading
        speed = np.sqrt(self.velocity[0] ** 2 + self.velocity[1] ** 2)
        heading_command = np.arcsin(np.clip(lateral_velocity_command / utils.not_zero(speed), -1, 1))
        heading_ref = np.clip(heading_command, -np.pi / 4, np.pi / 4)

        lane_heading = self.lane.get_heading_at(self.s)
        # Heading control
        # heading_rate_command = self.KP_HEADING * utils.wrap_to_pi(heading_ref + lane_heading - self.heading)
        heading_rate_command = utils.wrap_to_pi(heading_ref + lane_heading - self.heading) / dt

        # Heading rate to steering angle
        steering_angle = np.arctan(self.LENGTH / utils.not_zero(speed) * heading_rate_command)
        steering_angle = np.clip(steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)

        return steering_angle

    def velocity_control(self, step: int, dt: Optional[float] = None, trajectory: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Control the velocity of the vehicle.

        Using a simple proportional controller.

        :param step: the index of the trajectory
        :param dt: the frequency in the dataset
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
        self.velocity_history.append(self.velocity.copy())
        self.crash_history.append(self.crashed)
        super(HumanLikeVehicle, self).step(dt)

        self.traj.append([self.position.copy(), self.velocity.copy(), self.acceleration.copy(), self.heading])

    def calculate_human_likeness(self):
        original_traj = self.dataset_traj[:self.sim_steps + 1, :2]
        ego_traj = [trj[0] for trj in self.traj]
        ADE = np.mean([np.linalg.norm(original_traj[i] - ego_traj[i]) for i in
                       range(len(ego_traj))])  # Average Displacement Error (ADE)
        FDE = np.linalg.norm(original_traj[-1] - ego_traj[-1])  # Final Displacement Error (FDE)

        return FDE

from __future__ import division, print_function
import numpy as np
import copy
from typing import List

from sim4ad.irlenv import utils
from sim4ad.irlenv.vehicle.dynamics import Vehicle


class ControlledVehicle(Vehicle):
    """
    A vehicle piloted by two low-level controller, allowing high-level actions such as cruise control and lane changes.

    - The longitudinal controller is a velocity controller;
    - The lateral controller is a heading controller cascaded with a lateral position controller.
    """

    TAU_A = 0.6  # [s]
    TAU_DS = 0.2  # [s]
    TAU_LATERAL = 3  # [s]

    PURSUIT_TAU = 0.5 * TAU_DS  # [s]
    KP_A = 1 / TAU_A
    KP_HEADING = 1 / TAU_DS
    KP_LATERAL = 1 / TAU_LATERAL  # [1/s]
    MAX_STEERING_ANGLE = np.pi / 3  # [rad]
    DELTA_VELOCITY = 2  # [m/s]

    def __init__(self,
                 scenario_map,
                 position,
                 heading,
                 velocity,
                 target_lane=None,
                 target_velocity=None,
                 route=None):
        super(ControlledVehicle, self).__init__(scenario_map, position, heading, velocity)
        self.target_lane = target_lane or self.lane
        self.target_velocity = target_velocity or self.velocity
        self.route = route

    @classmethod
    def create_from(cls, vehicle):
        """
        Create a new vehicle from an existing one.
        The vehicle dynamics and target dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        v = cls(vehicle.scenario_map, vehicle.position, heading=vehicle.heading, velocity=vehicle.velocity,
                target_lane=vehicle.target_lane, target_velocity=vehicle.target_velocity, route=vehicle.route)
        return v

    def get_adjacent_target_lane(self, lane_change_left=False):
        """ depending on which lane change direction is allowed, we determine the target lane"""
        if lane_change_left:
            if self.lane.id < 0:
                target_lane = self.scenario_map.get_lane(self.lane.parent_road.id, self.lane.id + 1,
                                                         self.lane.lane_section.idx)
            else:
                target_lane = self.scenario_map.get_lane(self.lane.parent_road.id, self.lane.id - 1,
                                                         self.lane.lane_section.idx)
        else:
            if self.lane.id < 0:
                target_lane = self.scenario_map.get_lane(self.lane.parent_road.id, self.lane.id - 1,
                                                         self.lane.lane_section.idx)
            else:
                target_lane = self.scenario_map.get_lane(self.lane.parent_road.id, self.lane.id + 1,
                                                         self.lane.lane_section.idx)

        return target_lane if target_lane.type == 'driving' else None

    def act(self, action=None):
        """
        Perform a high-level action to change the desired lane or velocity.

        - If a high-level action is provided, update the target velocity and lane;
        - then, perform longitudinal and lateral control.

        :param action: a high-level action
        """

        if action == "FASTER":
            self.target_velocity += self.DELTA_VELOCITY

        elif action == "SLOWER":
            self.target_velocity -= self.DELTA_VELOCITY

        elif action == "LANE_RIGHT":
            self.target_lane = self.get_adjacent_target_lane

        elif action == "LANE_LEFT":
            self.target_lane = self.get_adjacent_target_lane(lane_change_left=True)

        action = {'steering': self.steering_control(self.target_lane),
                  'acceleration': self.velocity_control(self.target_velocity)}
        super(ControlledVehicle, self).act(action)

    def steering_control(self, target_lane):
        """
        Steer the vehicle to follow the center of a given lane.

        1. Lateral position is controlled by a proportional controller yielding a lateral velocity command
        2. Lateral velocity command is converted to a heading reference
        3. Heading is controlled by a proportional controller yielding a heading rate command
        4. Heading rate command is converted to a steering angle

        :param target_lane: the target lane to follow
        :return: a steering wheel angle command [rad]
        """
        target_coords = utils.local2frenet(point=self.position, reference_line=target_lane.midline)
        lane_next_coords = target_coords + self.velocity * self.PURSUIT_TAU
        lane_future_heading = target_lane.get_heading_at(lane_next_coords[0])

        # Lateral position control
        lateral_velocity_command = - self.KP_LATERAL * target_coords[1]

        # Lateral velocity to heading
        speed = np.sqrt(self.velocity[0] ** 2 + self.velocity[1] ** 2)
        heading_command = np.arcsin(np.clip(lateral_velocity_command / utils.not_zero(speed), -1, 1))
        heading_ref = lane_future_heading + np.clip(heading_command, -np.pi / 4, np.pi / 4)

        # Heading control
        heading_rate_command = self.KP_HEADING * utils.wrap_to_pi(heading_ref - self.heading)

        # Heading rate to steering angle
        steering_angle = np.arctan(self.LENGTH / utils.not_zero(speed) * heading_rate_command)
        steering_angle = np.clip(steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)

        return steering_angle

    def velocity_control(self, target_velocity):
        """
        Control the velocity of the vehicle.

        Using a simple proportional controller.

        :param target_velocity: the desired velocity
        :return: an acceleration command [m/s2]
        """
        return self.KP_A * (target_velocity - self.velocity)

    def predict_trajectory_constant_velocity(self, times, delta_t=0.033366700033366704):
        """
        Predict the future positions of the vehicle along its planned route, under constant velocity
        :param delta_t: the frequency of the dataset
        :param times: time steps of prediction
        :return: positions, headings
        """
        traj = []
        position = self.position
        for t in times:
            s, _ = utils.local2frenet(point=position, reference_line=self.lane.midline)
            heading = self.lane.get_heading_at(self.s)
            rotation_matrix = np.array([
                [np.cos(heading), -np.sin(heading)],
                [np.sin(heading), np.cos(heading)]
            ])
            position += np.dot(rotation_matrix, self.velocity) * delta_t * t
            traj.append(position)

        return traj


class MDPVehicle(ControlledVehicle):
    """
    A controlled vehicle with a specified discrete range of allowed target velocities.
    """

    SPEED_COUNT = 21  # []
    SPEED_MIN = 0  # [m/s]
    SPEED_MAX = 20  # [m/s]

    def __init__(self,
                 scenario_map,
                 position,
                 heading,
                 velocity,
                 target_lane=None,
                 target_velocity=None,
                 route=None,
                 dataset_traj=None,
                 vehicle_id=None, v_length=None, v_width=None):
        super(MDPVehicle, self).__init__(scenario_map, position, heading, velocity, target_lane, target_velocity, route)
        self.velocity_index = self.speed_to_index(self.target_velocity)
        self.target_velocity = self.index_to_speed(self.velocity_index)
        self.dataset_traj = dataset_traj
        self.traj = np.array(self.position)
        self.sim_steps = 0
        self.vehicle_id = vehicle_id
        self.LENGTH = v_length  # Vehicle length [m]
        self.WIDTH = v_width  # Vehicle width [m]

    @classmethod
    def create(cls, scenario_map, vehicle_id, position, v_length, v_width, dataset_traj, heading: float,
               velocity: List[float],
               target_velocity):
        """
        Create a human-like driving vehicle in replace of a dataset vehicle.

        :param target_velocity:
        :param heading:
        :param vehicle_id:
        :param dataset_traj:
        :param v_width:
        :param v_length:
        :param scenario_map: the road where the vehicle is driving
        :param position: the position where the vehicle start on the road
        :param velocity: initial velocity in [m/s]. If None, will be chosen randomly
        :return: A vehicle with random position and/or velocity
        """
        v = cls(scenario_map, position, heading, velocity, target_velocity=target_velocity,
                vehicle_id=vehicle_id, v_length=v_length, v_width=v_width, dataset_traj=dataset_traj)

        return v

    def act(self, action=None):
        """
        Perform a high-level action.

        If the action is a velocity change, choose velocity from the allowed discrete range.
        Else, forward action to the ControlledVehicle handler.

        :param action: a high-level action
        """
        if action == "FASTER":
            self.velocity_index = self.speed_to_index(self.velocity) + 1
        elif action == "SLOWER":
            self.velocity_index = self.speed_to_index(self.velocity) - 1
        else:
            super(MDPVehicle, self).act(action)
            return
        # print(self.velocity_index)
        self.velocity_index = np.clip(self.velocity_index, 0, self.SPEED_COUNT - 1)
        self.target_velocity = self.index_to_speed(self.velocity_index)
        super().act()

    def step(self, dt):
        self.sim_steps += 1
        super(MDPVehicle, self).step(dt)

        self.traj = np.append(self.traj, self.position, axis=0)

    @classmethod
    def index_to_speed(cls, index):
        """
        Convert an index among allowed speeds to its corresponding speed
        :param index: the speed index []
        :return: the corresponding speed [m/s]
        """
        if cls.SPEED_COUNT > 1:
            return cls.SPEED_MIN + index * (cls.SPEED_MAX - cls.SPEED_MIN) / (cls.SPEED_COUNT - 1)
        else:
            return cls.SPEED_MIN

    @classmethod
    def speed_to_index(cls, speed):
        """
        Find the index of the closest speed allowed to a given speed.
        :param speed: an input speed [m/s]
        :return: the index of the closest speed allowed []
        """
        x = (speed - cls.SPEED_MIN) / (cls.SPEED_MAX - cls.SPEED_MIN)
        return np.int(np.clip(np.round(x * (cls.SPEED_COUNT - 1)), 0, cls.SPEED_COUNT - 1))

    def speed_index(self):
        """
        The index of current velocity
        """
        return self.speed_to_index(self.velocity)

    def predict_trajectory(self, actions, action_duration, trajectory_timestep, dt):
        """
        Predict the future trajectory of the vehicle given a sequence of actions.

        :param actions: a sequence of future actions.
        :param action_duration: the duration of each action.
        :param trajectory_timestep: the duration between each save of the vehicle state.
        :param dt: the timestep of the simulation
        :return: the sequence of future states
        """
        states = []
        v = copy.deepcopy(self)
        t = 0
        for action in actions:
            v.act(action)  # High-level decision
            for _ in range(int(action_duration / dt)):
                t += 1
                v.act()  # Low-level control action
                v.step(dt)
                if (t % int(trajectory_timestep / dt)) == 0:
                    states.append(copy.deepcopy(v))

        return states

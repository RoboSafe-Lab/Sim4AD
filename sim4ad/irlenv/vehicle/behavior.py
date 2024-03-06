from __future__ import division, print_function
import numpy as np
from typing import Tuple

from sim4ad.irlenv.vehicle.control import ControlledVehicle
from sim4ad.irlenv import utils

np.random.seed(42)


class IDMVehicle(ControlledVehicle):
    """
    A vehicle using both a longitudinal and a lateral decision policies.

    - Longitudinal: the IDM model computes an acceleration given the preceding vehicle's distance and velocity.
    - Lateral: the MOBIL model decides when to change lane by maximizing the acceleration of nearby vehicles.
    """

    # Longitudinal policy parameters
    ACC_MAX = 1.5  # [m/s2]
    COMFORT_ACC_MAX = 0.7  # [m/s2]
    COMFORT_ACC_MIN = -0.7  # [m/s2]
    DISTANCE_WANTED = 1.5  # [m]
    TIME_WANTED = 1.2  # [s]
    DELTA = 4.0  # []

    # Lateral policy parameters
    POLITENESS = 0.01  # in [0, 1]
    LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0  # [m/s2]
    LANE_CHANGE_DELAY = 1.0  # [s]

    def __init__(self, scenario_map, position,
                 heading,
                 velocity,
                 target_lane=None,
                 target_velocity=None,
                 route=None,
                 enable_lane_change=True,
                 timer=None,
                 vehicle_id=None):
        super(IDMVehicle, self).__init__(scenario_map, position, heading, velocity, target_lane, target_velocity, route)
        self.enable_lane_change = enable_lane_change
        self.timer = timer or (np.sum(self.position) * np.pi) % self.LANE_CHANGE_DELAY
        self.vehicle_id = vehicle_id

    def randomize_behavior(self):
        pass

    @classmethod
    def create_from(cls, vehicle):
        """
        Create a new vehicle from an existing one.
        The vehicle dynamics and target dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        v = cls(vehicle.scenario_map, vehicle.position, heading=vehicle.heading, velocity=vehicle.velocity,
                target_lane=vehicle.target_lane, target_velocity=vehicle.target_velocity,
                route=vehicle.route, timer=getattr(vehicle, 'timer', None))

        return v

    @staticmethod
    def get_front_rear_vehicle(vehicles, subject, target_lane=None) -> Tuple[tuple, tuple]:
        """Get the front and rear vehicle of ego at the target lane"""
        dis_front = np.inf
        dis_rear = -np.inf
        front_vehicle = rear_vehicle = None
        if target_lane is None:
            target_lane = subject.lane
        for vehicle in vehicles:
            if vehicle.vehicle_id == subject.vehicle_id:
                continue
            if vehicle.lane == target_lane:
                # check the s value along the lane
                s, d = utils.local2frenet(vehicle.position, target_lane.midline)
                if s > subject.s and s - subject.s < dis_front:
                    dis_front = s - subject.s
                    front_vehicle = vehicle
                elif s < subject.s and s - subject.s > dis_rear:
                    dis_rear = s - subject.s
                    rear_vehicle = vehicle

        return (front_vehicle, dis_front), (rear_vehicle, dis_rear)

    def act(self, active_vehicles=None, action=None):
        """
        Execute an action.

        For now, no action is supported because the vehicle takes all decisions
        of acceleration and lane changes on its own, based on the IDM and MOBIL models.

        :param active_vehicles:
        :param action: the action
        """
        if self.crashed:
            return
        action = {}
        front_vehicle, rear_vehicle = self.get_front_rear_vehicle(active_vehicles, self)

        # Lateral: MOBIL
        if self.enable_lane_change:
            self.change_lane_policy()
        action['steering'] = self.steering_control(self.target_lane)

        # Longitudinal: IDM
        action['acceleration'] = self.acceleration(ego_vehicle=self, front_vehicle=front_vehicle,
                                                   rear_vehicle=rear_vehicle)
        action['acceleration'] = np.clip(action['acceleration'], -self.ACC_MAX, self.ACC_MAX)

        super(ControlledVehicle, self).act(action)

    def step(self, dt):
        """
        Step the simulation.

        Increases a timer used for decision policies, and step the vehicle dynamics.

        :param dt: timestep
        """
        self.timer += dt
        super(IDMVehicle, self).step(dt)

    def acceleration(self, ego_vehicle, front_vehicle=None, rear_vehicle=None):
        """
        Compute an acceleration command with the Intelligent Driver Model.

        The acceleration is chosen so as to:
        - reach a target velocity;
        - maintain a minimum safety distance (and safety time) w.r.t the front vehicle.

        :param ego_vehicle: the vehicle whose desired acceleration is to be computed. It does not have to be an
                            IDM vehicle, which is why this method is a class method. This allows an IDM vehicle to
                            reason about other vehicles behaviors even though they may not IDMs.
        :param front_vehicle: the vehicle preceding the ego-vehicle
        :param rear_vehicle: the vehicle following the ego-vehicle
        :return: the acceleration command for the ego-vehicle [m/s2]
        """
        if not ego_vehicle:
            return 0

        ego_velocity = getattr(ego_vehicle, "target_velocity", 0)
        ego_target_velocity_x = utils.not_zero(ego_velocity[0])
        acceleration = self.COMFORT_ACC_MAX * (
                    1 - np.power(max(ego_vehicle.velocity[0], 0) / ego_target_velocity_x, self.DELTA))

        if front_vehicle[0]:
            d = front_vehicle[1]
            acceleration -= self.COMFORT_ACC_MAX * np.power(
                self.desired_gap(ego_vehicle, front_vehicle[0]) / utils.not_zero(d), 2)

        return acceleration

    def desired_gap(self, ego_vehicle, front_vehicle=None):
        """
        Compute the desired distance between a vehicle and its leading vehicle.

        :param ego_vehicle: the vehicle being controlled
        :param front_vehicle: its leading vehicle
        :return: the desired distance between the two vehicles
        """
        d0 = self.DISTANCE_WANTED + ego_vehicle.LENGTH / 2 + front_vehicle.LENGTH / 2
        tau = self.TIME_WANTED
        ab = -self.COMFORT_ACC_MAX * self.COMFORT_ACC_MIN
        dv = ego_vehicle.velocity[0] - front_vehicle.velocity[0]
        d_star = d0 + ego_vehicle.velocity[0] * tau + ego_vehicle.velocity[0] * dv / (2 * np.sqrt(ab))

        return d_star

    def maximum_velocity(self, front_vehicle=None):
        """
        Compute the maximum allowed velocity to avoid Inevitable Collision States.

        Assume the front vehicle is going to brake at full deceleration and that
        it will be noticed after a given delay, and compute the maximum velocity
        which allows the ego-vehicle to brake enough to avoid the collision.

        :param front_vehicle: the preceding vehicle
        :return: the maximum allowed velocity, and suggested acceleration
        """

        if not front_vehicle:
            return self.target_velocity
        d0 = self.DISTANCE_WANTED
        a0 = self.COMFORT_ACC_MIN
        a1 = self.COMFORT_ACC_MIN
        tau = self.TIME_WANTED
        d = max(self.lane_distance_to(front_vehicle) - self.LENGTH / 2 - front_vehicle.LENGTH / 2 - d0, 0)
        v1_0 = front_vehicle.velocity
        delta = 4 * (a0 * a1 * tau) ** 2 + 8 * a0 * (a1 ** 2) * d + 4 * a0 * a1 * v1_0 ** 2
        v_max = -a0 * tau + np.sqrt(delta) / (2 * a1)

        # Velocity control
        self.target_velocity = min(self.maximum_velocity(front_vehicle), self.target_velocity)
        acceleration = self.velocity_control(self.target_velocity)

        return v_max, acceleration

    def change_lane_policy(self, active_vehicles=None):
        """
        Decide when to change lane.

        Based on:
        - frequency;
        - closeness of the target lane;
        - MOBIL model.
        """
        # If a lane change already ongoing
        if self.lane != self.target_lane:
            # Abort it if someone else is already changing into the same lane with small distance
            for v in active_vehicles:
                if v is not self \
                        and v.lane != self.target_lane \
                        and isinstance(v, ControlledVehicle) \
                        and v.target_lane == self.target_lane:
                    d = v.s - self.s
                    d_star = self.desired_gap(self, v)
                    if 0 < d < d_star:
                        self.target_lane = self.lane
                        break
            return

        # else, at a given frequency,
        if not utils.do_every(self.LANE_CHANGE_DELAY, self.timer):
            return
        self.timer = 0

        # decide to make a lane change, only lane change left is considered
        left_lane = self.get_adjacent_target_lane(lane_change_left=True)
        # Does the MOBIL model recommend a lane change?
        if self.mobil(active_vehicles, left_lane):
            self.target_lane = left_lane

    def mobil(self, active_vehicles, lane):
        """
        MOBIL lane change model: Minimizing Overall Braking Induced by a Lane change

        The vehicle should change lane only if:
        - after changing it (and/or following vehicles) can accelerate more;
        - it doesn't impose an unsafe braking on its new following vehicle.

        :param active_vehicles:
        :param lane: the candidate lane for the change
        :return: whether the lane change should be performed
        """
        # Is the maneuver unsafe for the new following vehicle?
        new_preceding, new_following = self.get_front_rear_vehicle(active_vehicles, self, lane)
        new_following_a = self.acceleration(ego_vehicle=new_following[0], front_vehicle=new_preceding)
        front_vehicle = (self, new_following[1])
        new_following_pred_a = self.acceleration(ego_vehicle=new_following[0], front_vehicle=front_vehicle)

        if new_following_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
            return False

        # Do I have a planned route for a specific lane which is safe for me to access?
        old_preceding, old_following = self.get_front_rear_vehicle(active_vehicles, self)
        self_pred_a = self.acceleration(ego_vehicle=self, front_vehicle=new_preceding)

        # Unsafe braking required
        if self_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
            return False

        # Is there an acceleration advantage for me and/or my followers to change lane?
        else:
            self_a = self.acceleration(ego_vehicle=self, front_vehicle=old_preceding)
            front_vehicle = (self, old_following[1])
            old_following_a = self.acceleration(ego_vehicle=old_following[0], front_vehicle=front_vehicle)
            old_following_pred_a = self.acceleration(ego_vehicle=old_following[0], front_vehicle=old_preceding)
            jerk = self_pred_a - self_a + self.POLITENESS * (
                        new_following_pred_a - new_following_a + old_following_pred_a - old_following_a)
            if jerk < self.LANE_CHANGE_MIN_ACC_GAIN:
                return False

        # All clear, let's go!
        return True


class LinearVehicle(IDMVehicle):
    """
    A Vehicle whose longitudinal and lateral controllers are linear with respect to parameters
    """
    ACCELERATION_PARAMETERS = [0.3, 0.14, 0.8]
    STEERING_PARAMETERS = [ControlledVehicle.KP_HEADING, ControlledVehicle.KP_HEADING * ControlledVehicle.KP_LATERAL]

    ACCELERATION_RANGE = np.array([0.5 * np.array(ACCELERATION_PARAMETERS), 1.5 * np.array(ACCELERATION_PARAMETERS)])
    STEERING_RANGE = np.array([np.array(STEERING_PARAMETERS) - np.array([0.07, 1.5]),
                               np.array(STEERING_PARAMETERS) + np.array([0.07, 1.5])])

    TIME_WANTED = 2.0

    def __init__(self, road, position,
                 heading=0,
                 velocity=0,
                 target_lane_index=None,
                 target_velocity=None,
                 route=None,
                 enable_lane_change=True,
                 timer=None):
        super(LinearVehicle, self).__init__(road,
                                            position,
                                            heading,
                                            velocity,
                                            target_lane_index,
                                            target_velocity,
                                            route,
                                            enable_lane_change,
                                            timer)

    def randomize_behavior(self):
        ua = np.random.uniform(size=np.shape(self.ACCELERATION_PARAMETERS))
        self.ACCELERATION_PARAMETERS = self.ACCELERATION_RANGE[0] + ua * (
                    self.ACCELERATION_RANGE[1] - self.ACCELERATION_RANGE[0])
        ub = np.random.uniform(size=np.shape(self.STEERING_PARAMETERS))
        self.STEERING_PARAMETERS = self.STEERING_RANGE[0] + ub * (self.STEERING_RANGE[1] - self.STEERING_RANGE[0])

    def acceleration(self, ego_vehicle, front_vehicle=None, rear_vehicle=None):
        """
        Compute an acceleration command with a Linear Model.

        The acceleration is chosen so as to:
        - reach a target velocity;
        - reach the velocity of the leading (resp following) vehicle, if it is lower (resp higher) than ego's;
        - maintain a minimum safety distance w.r.t the leading vehicle.

        :param ego_vehicle: the vehicle whose desired acceleration is to be computed. It does not have to be an
                            Linear vehicle, which is why this method is a class method. This allows a Linear vehicle to
                            reason about other vehicles behaviors even though they may not Linear.
        :param front_vehicle: the vehicle preceding the ego-vehicle
        :param rear_vehicle: the vehicle following the ego-vehicle
        :return: the acceleration command for the ego-vehicle [m/s2]
        """
        return np.dot(self.ACCELERATION_PARAMETERS,
                      self.acceleration_features(ego_vehicle, front_vehicle))

    def acceleration_features(self, ego_vehicle, front_vehicle=None):
        vt, dv, dp = 0, 0, 0
        if ego_vehicle:
            vt = ego_vehicle.target_velocity - ego_vehicle.velocity
            d_safe = self.DISTANCE_WANTED + np.max(ego_vehicle.velocity, 0) * self.TIME_WANTED + ego_vehicle.LENGTH
            if front_vehicle:
                d = ego_vehicle.lane_distance_to(front_vehicle)
                dv = min(front_vehicle.velocity - ego_vehicle.velocity, 0)
                dp = min(d - d_safe, 0)
        return np.array([vt, dv, dp])

    def steering_control(self, target_lane):
        """
        Linear controller with respect to parameters.
        Overrides the non-linear controller ControlledVehicle.steering_control()
        :param target_lane: index of the lane to follow
        :return: a steering wheel angle command [rad]
        """
        steering_angle = np.dot(np.array(self.STEERING_PARAMETERS), self.steering_features(target_lane))
        steering_angle = np.clip(steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
        return steering_angle

    def steering_features(self, target_lane):
        """
        A collection of features used to follow a lane
        :param target_lane: the target lane to follow
        :return: an array of features
        """
        lane = target_lane
        lane_coords = lane.local_coordinates(self.position)
        lane_next_coords = lane_coords[0] + self.velocity * self.PURSUIT_TAU
        lane_future_heading = lane.heading_at(lane_next_coords)
        features = np.array([utils.wrap_to_pi(lane_future_heading - self.heading) *
                             self.LENGTH / utils.not_zero(self.velocity),
                             -lane_coords[1] * self.LENGTH / (utils.not_zero(self.velocity) ** 2)])
        return features


class AggressiveVehicle(LinearVehicle):
    LANE_CHANGE_MIN_ACC_GAIN = 1.0  # [m/s2]
    MERGE_ACC_GAIN = 0.8
    MERGE_VEL_RATIO = 0.75
    MERGE_TARGET_VEL = 30
    ACCELERATION_PARAMETERS = [MERGE_ACC_GAIN / ((1 - MERGE_VEL_RATIO) * MERGE_TARGET_VEL),
                               MERGE_ACC_GAIN / (MERGE_VEL_RATIO * MERGE_TARGET_VEL),
                               0.5]


class DefensiveVehicle(LinearVehicle):
    LANE_CHANGE_MIN_ACC_GAIN = 1.0  # [m/s2]
    MERGE_ACC_GAIN = 1.2
    MERGE_VEL_RATIO = 0.75
    MERGE_TARGET_VEL = 30
    ACCELERATION_PARAMETERS = [MERGE_ACC_GAIN / ((1 - MERGE_VEL_RATIO) * MERGE_TARGET_VEL),
                               MERGE_ACC_GAIN / (MERGE_VEL_RATIO * MERGE_TARGET_VEL),
                               2.0]

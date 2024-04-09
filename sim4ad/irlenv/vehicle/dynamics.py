from __future__ import division, print_function

from loguru import logger

import numpy as np
import pandas as pd
from collections import deque
from typing import List

from sim4ad.irlenv import utils
from sim4ad.irlenv.logger import Loggable


class Vehicle(Loggable):
    """
    A moving vehicle on a road, and its dynamics.

    The vehicle is represented by a dynamical system: a modified bicycle model.
    It is state is propagated depending on its steering and acceleration actions.
    """
    # Enable collision detection between vehicles 
    COLLISIONS_ENABLED = True
    # Vehicle length [m]
    LENGTH = 5.0
    # Vehicle width [m]
    WIDTH = 2.0
    # Range for random initial velocities [m/s]
    DEFAULT_VELOCITIES = [23, 25]
    # Maximum reachable velocity [m/s]
    MAX_VELOCITY = 50

    def __init__(self, scenario_map, position: List[float], heading: float, velocity:  List[float]):
        self.scenario_map = scenario_map
        self.position = np.array(position)
        self.heading = heading
        self.velocity = np.array(velocity)
        self.lane = scenario_map.best_lane_at(point=position, heading=heading)
        self.action = {'steering': 0.0, 'acceleration': np.array([0.0, 0.0])}
        self.crashed = False
        self.log = []
        self.history = deque(maxlen=50)
        self.record_history = False
        if self.lane is None:
            raise AttributeError(f'lane is None for position: {position}, heading: {heading}.')
        self.s, self.d = utils.local2frenet(point=self.position, reference_line=self.lane.midline)

    @classmethod
    def make_on_lane(cls, scenario_map, road_id, lane_id, longitudinal, lateral, velocity: List[float]):
        """
        Create a vehicle on a given lane at a longitudinal position.

        :param lateral: lateral position along the lane
        :param road_id: the id of the road
        :param scenario_map: the map where the vehicle is driving
        :param lane_id: index of the lane where the vehicle is located
        :param longitudinal: longitudinal position along the lane
        :param velocity: initial velocity in [m/s]
        :return: A vehicle with at the specified position
        """
        lane = scenario_map.get_lane(road_id=road_id, lane_id=lane_id)

        pos_x, pos_y = utils.frenet2local(reference_line=lane.midline, s=longitudinal, d=lateral)
        lane_heading = lane.get_heading_at(longitudinal)

        if velocity is None:
            velocity = [cls.MAX_VELOCITY, 0]

        return cls(scenario_map, [pos_x, pos_y], lane_heading, velocity)

    @classmethod
    def create_random(cls, road, velocity=None, spacing=1):
        """
        Create a random vehicle on the road.

        The lane and /or velocity are chosen randomly, while longitudinal position is chosen behind the last
        vehicle in the road with density based on the number of lanes.

        :param road: the road where the vehicle is driving
        :param velocity: initial velocity in [m/s]. If None, will be chosen randomly
        :param spacing: ratio of spacing to the front vehicle, 1 being the default
        :return: A vehicle with random position and/or velocity
        """
        if velocity is None:
            velocity = road.np_random.uniform(Vehicle.DEFAULT_VELOCITIES[0], Vehicle.DEFAULT_VELOCITIES[1])

        default_spacing = 1.5*velocity

        _from = road.np_random.choice(list(road.network.graph.keys()))
        _to = road.np_random.choice(list(road.network.graph[_from].keys()))
        _id = road.np_random.choice(len(road.network.graph[_from][_to]))

        offset = spacing * default_spacing * np.exp(-5 / 30 * len(road.network.graph[_from][_to]))
        x0 = np.max([v.position[0] for v in road.vehicles]) if len(road.vehicles) else 3*offset
        x0 += offset * road.np_random.uniform(0.9, 1.1)
        
        v = cls(road,
                road.network.get_lane((_from, _to, _id)).position(x0, 0),
                road.network.get_lane((_from, _to, _id)).heading_at(x0),
                velocity)

        return v

    @classmethod
    def create_from(cls, vehicle):
        """
        Create a new vehicle from an existing one.
        Only the vehicle dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        v = cls(vehicle.scenario_map, vehicle.position, vehicle.heading, vehicle.velocity)

        return v

    def act(self, action=None):
        """
        Store an action to be repeated.

        :param action: the input action
        """
        if action:
            self.action = action

    def step(self, dt):
        """
        Propagate the vehicle state given its actions.

        Integrate a modified bicycle model with a 1st-order response on the steering wheel dynamics.
        If the vehicle is crashed, the actions are overridden with erratic steering and braking until complete stop.
        The vehicle's current lane is updated.

        :param dt: timestep of integration of the model [s]
        """
        if self.crashed:
            self.action['steering'] = 0
            self.action['acceleration'] = -1.0*self.velocity

        speed = np.sqrt(self.velocity[0] ** 2 + self.velocity[1] ** 2)
        if speed > self.MAX_VELOCITY:
            self.action['acceleration'] = np.array([min(self.action['acceleration'][0], 1.0*(self.MAX_VELOCITY - speed)),
                                                    self.action['acceleration'][1]])

        # update position
        rotation_matrix = np.array([
            [np.cos(self.heading), -np.sin(self.heading)],
            [np.sin(self.heading), np.cos(self.heading)]
        ])
        self.position += np.dot(rotation_matrix, self.velocity) * dt
        self.heading += speed * np.tan(self.action['steering']) / self.LENGTH * dt
        delta_v = self.action['acceleration'] * dt
        self.velocity += delta_v

        if self.scenario_map:
            lane = self.scenario_map.best_lane_at(point=self.position, heading=self.heading)
            # lane can not be determined, for example reaching road end
            if lane is not None:
                self.lane = lane
            else:
                logging.warning(f'No lane found at position {self.position} and heading {self.heading}.')
            self.s, self.d = utils.local2frenet(point=self.position, reference_line=self.lane.midline)
            if self.record_history:
                self.history.appendleft(self.create_from(self))

    def lane_distance_to(self, vehicle):
        """
        Compute the signed distance to another vehicle along current lane.

        :param vehicle: the other vehicle
        :return: the distance to the other vehicle [m]
        """
        if not vehicle:
            return np.nan
        
        return self.lane.local_coordinates(vehicle.position)[0] - self.lane.local_coordinates(self.position)[0]

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

        # Accurate rectangular check
        if utils.rotated_rectangles_intersect((self.position, 0.9*self.LENGTH, 0.9*self.WIDTH, self.heading),
                                              (other.position, 0.9*other.LENGTH, 0.9*other.WIDTH, other.heading)):
            if self.velocity[0] <= other.velocity[0]:
                self.velocity = other.velocity = self.velocity
            else:
                self.velocity = other.velocity = other.velocity
            self.crashed = other.crashed = True

    @property
    def direction(self):
        return np.array([np.cos(self.heading), np.sin(self.heading)])

    @property
    def destination(self):
        if getattr(self, "route", None):
            last_lane = self.lane
            return last_lane.position(last_lane.length, 0)
        else:
            return self.position

    @property
    def destination_direction(self):
        if (self.destination != self.position).any():
            return (self.destination - self.position) / np.linalg.norm(self.destination - self.position)
        else:
            return np.zeros((2,))

    @property
    def on_road(self):
        """ Is the vehicle on its current lane, or off-road ? """
        return True if self.lane is not None else False

    def front_distance_to(self, other):
        return self.direction.dot(other.position - self.position)

    def to_dict(self, origin_vehicle=None, observe_intentions=True):
        d = {
            'presence': 1,
            'x': self.position[0],
            'y': self.position[1],
            'vx': self.velocity * self.direction[0],
            'vy': self.velocity * self.direction[1],
            'cos_h': self.direction[0],
            'sin_h': self.direction[1],
            'cos_d': self.destination_direction[0],
            'sin_d': self.destination_direction[1]
        }
        if not observe_intentions:
            d["cos_d"] = d["sin_d"] = 0
        if origin_vehicle:
            origin_dict = origin_vehicle.to_dict()
            for key in ['x', 'y', 'vx', 'vy']:
                d[key] -= origin_dict[key]

        return d

    def dump(self):
        """
        Update the internal log of the vehicle, containing:
        - its kinematics;
        - some metrics relative to its neighbour vehicles.
        """
        data = {
            'x': self.position[0],
            'y': self.position[1],
            'psi': self.heading,
            'vx': self.velocity * np.cos(self.heading),
            'vy': self.velocity * np.sin(self.heading),
            'v': self.velocity,
            'acceleration': self.action['acceleration'],
            'steering': self.action['steering']}

        self.log.append(data)

    def get_log(self):
        """
        Cast the internal log as a DataFrame.

        :return: the DataFrame of the Vehicle's log.
        """
        return pd.DataFrame(self.log)

    def __str__(self):
        return "{} #{}: {}".format(self.__class__.__name__, id(self) % 1000, self.position)

    def __repr__(self):
        return self.__str__()


class Obstacle(Vehicle):
    """
    A motionless obstacle at a given position.
    """

    def __init__(self, road, position, heading=0):
        super(Obstacle, self).__init__(road, position, velocity=[0, 0], heading=heading)
        self.target_velocity = 0
        self.LENGTH = self.WIDTH

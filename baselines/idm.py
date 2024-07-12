"""
Model a Intelligent Driver Model (IDM) baseline.
"""
from typing import Any

import numpy as np

from simulator.state_action import State, Observation, Action


class IDM:

    # Longitudinal policy parameters based on what Cheng suggested
    a = 5  # max acceleration [m/s2]
    b = 0.7  # comfortable braking deceleration [m/s2]
    s0 = 2   # minimum gap [m]
    T = 1.5  # desired headway [s]
    DELTA = 4.0  # [] - acceleration exponent

    def __init__(self):
        """
        :param v0: desired velocity in free traffic.
        """
        self.v0 = None

    def compute_idm_action(self, state_i, agent_in_front, agent_meta, previous_state_i) -> Action:
        """
        Compute the acceleration of the vehicle.
        :param state_i: state of i
        :param agent_in_front: vehicle in front of vehicle i
        """

        assert self.v0 is not None, "IDM is not activated"

        v = state_i.speed
        xi = state_i.position

        if agent_in_front is None:
            acc = self.a * (1 - np.power(v / self.v0, self.DELTA))
            acc = np.clip(acc, -self.a, self.a)
            return Action(acceleration=float(acc), steer_angle=0.0)

        x_front = agent_in_front["position"]
        v_front = agent_in_front["speed"]
        l_front = agent_in_front["metadata"].length

        actual_gap = xi.distance(x_front) - l_front
        star_gap = self.desired_gap(v, v_front)
        acc = self.a * (1 - np.power(v / self.v0, self.DELTA) - np.power(star_gap / actual_gap, 2))

        acc = np.clip(acc, -self.a, self.a)

        return Action(acceleration=float(acc), steer_angle=0)

    def activated(self):
        return self.v0 is not None

    def activate(self, v0):
        self.v0 = v0

    def desired_gap(self, va, v_front):
        """
        Compute the desired gap between two vehicles.

        :param va: speed of the ego.
        :param v_front: speed of the leading vehicle.
        :return: comfortable braking deceleration.
        """
        delta_va = va - v_front
        return self.s0 + va * self.T + va * delta_va / (2 * np.sqrt(self.a * self.b))



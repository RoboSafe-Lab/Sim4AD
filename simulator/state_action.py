"""
Define the characteristics that a state (observation) and action at time t should have in the lightweight simulator.
"""

# Get the list of features used in the state from common_elements.json and create an object out of it
# to be used in the simulator.

import json
import os
from dataclasses import dataclass
from random import random
from typing import Dict, List, Tuple, Union, Any
from dataclasses import dataclass

import numpy as np
from shapely import Point

from sim4ad.agentstate import AgentMetadata
from sim4ad.opendrive import Lane
from sim4ad.path_utils import get_common_property
from sim4ad.util import Box


class ActionState:
    """
    Abstract class for either an action or an observation, where the attributes depend on the features included in the
    preprocessing/feature extraction stage.
    """

    def set_observation(self, type:str, state: Dict[str, float]):
        """
        type: str - Whether the class represent the 'state' or 'action'.
        state: Dict[str, float] - The state or action to set the features of the class to.
        """

        if type not in ['observation', 'action']:
            raise ValueError(f"Type must be either 'observation' or 'action', not {type}")

        # The option space is the set of features that can be used (as set in preprocessing.py)
        option_space_to_use = "FEATURES_IN_OBSERVATIONS" if type == "observation" else "FEATURES_IN_ACTIONS"
        self.__option_space = list(get_common_property(option_space_to_use))

        self.__features = {k: None for k in self.__option_space}

        for feature_name, value in state.items():
            self.__features[feature_name] = value

        # If there is a none value in the features, raise an error
        if None in self.__features.values():
            raise ValueError(f"Features are not set properly: {self.__features}")

    def get_tuple(self):
        """
        Return a tuple of the values of the features in the state, in the order set by the state space.
        """
        representation = tuple(self.__features.values())

        # Check that the value at a random index matches the value at the same index in the action space.
        # This is to ensure that the feature order is consistent with the training data in preprocessing.py
        i = int(random() * len(representation))
        assert representation[i] == self.__features[self.__option_space[i]], (f"Representation of state is not in the "
                                                                              f"correct order: {representation}")

        return representation

    def get_feature(self, feature_name: str) -> Any:
        """
        Return the value of the feature with the given name.
        """
        return self.__features[feature_name]


class State:
    """
    Class for a state in the simulator.
    """

    def __init__(self, time: float, position: Union[np.ndarray[float, float], Point], speed: float, acceleration: float,
                 heading: float, agent_length: float, agent_width: float, lane: Lane = None):
        self.time = time

        if isinstance(position, np.ndarray) or isinstance(position, list) or isinstance(position, tuple):
            self.position = Point(position[0], position[1])
        else:
            self.position = position
        self.speed = speed
        self.acceleration = acceleration
        self.heading = heading
        self.lane = lane

        center = np.array([self.position.x, self.position.y])
        self.bbox = Box(center=center, length=agent_length, width=agent_width, heading=heading)


class Observation(ActionState):
    """
    Class for an observation in the simulator.
    """

    def __init__(self, state: Dict[str, float]):
        super().__init__()
        self.set_observation(type="observation", state=state)


class Action:
    """
    Class for an action in the simulator.
    """

    def __init__(self, acceleration: float, steer_angle: float):
        self.__acceleration = acceleration
        self.__steer_angle = steer_angle

    @property
    def acceleration(self):
        return self.__acceleration

    @property
    def steer_angle(self):
        return self.__steer_angle


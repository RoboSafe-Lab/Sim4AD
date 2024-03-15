"""
Class to define an agent that follows a given policy pi(s) -> (acceleration, steering_angle).
"""
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

from simulator.state_action import Observation, Action, State
from sim4ad.agentstate import AgentMetadata

import logging

logger = logging.getLogger(__name__)


class PolicyAgent:
    def __init__(self, agent: Any, policy, initial_state: State, original_initial_time: float):
        """
        Initialize a new PolicyAgent with the given policy.

        :param agent: The agent as loaded from the episode.
        :param policy: A function that takes history of observations and returns acceleration and steering angle.

        :param original_initial_time: The time at which the corresponding agent was created in the DATASET.
        """

        self._agent_id = agent.UUID
        self._metadata = AgentMetadata(length=agent.length, width=agent.width, agent_type=agent.type,
                                       front_overhang=0.91, rear_overhang=1.094,  # TODO: front and rear overhang are arbitrary
                                       wheelbase=agent.length*0.6,  # TODO: arbitrary wheelbase
                                       max_acceleration=5.0,  # TODO: arbitrary max acceleration
                                       max_angular_vel=2.0  # TODO: arbitrary max angular velocity
                                       )

        self.policy = policy
        self._state_trajectory = [initial_state]
        self._obs_trajectory = []
        self._action_trajectory = []
        self.__evaluation_features_trajectory = defaultdict(list)  # List of all the features used for evaluation at each time step.
        self._initial_state = initial_state
        self._original_initial_time = original_initial_time

        # For the bicycle model
        correction = (self._metadata.rear_overhang - self._metadata.front_overhang) / 2  # Correction for cg
        self._l_f = self._metadata.wheelbase / 2 + correction  # Distance of front axel from cg
        self._l_r = self._metadata.wheelbase / 2 - correction  # Distance of back axel from cg

    def next_action(self, history: [[Observation]]) -> Action:
        """
        Get the action for the given state history.

        :param history: The history of the observations/states.
        :return: The desired acceleration and steering angle.
        """
        acceleration, delta = self.policy(history)[0].tolist()

        action = Action(acceleration=acceleration, steer_angle=delta)

        return action

    @staticmethod
    def reached_goal(state: State) -> bool:
        """
        Check if the agent has reached the goal.

        :param state: The current state of the agent.
        :return: Whether the agent has reached the goal.
        """

        if state.lane is None:
            # The agent went out of the road.
            return False

        reached_end_lane = state.lane.distance_at(state.position) > 0.97 * state.lane.length
        if reached_end_lane:
            # TODO: adapt for other scenarios
            logger.warning("This only works for AUTOMATUM where is there is only one lane")
        return reached_end_lane

    def __call__(self, history: [[Observation]]) -> tuple[Observation, Action]:
        """
        Get the action for the given state history.

        :param history: The history of the observations/states.
        :return: The desired acceleration and steering angle.
        """
        return self.next_action(history)

    def terminated(self, max_steps: int) -> bool:
        return len(self._state_trajectory) >= max_steps
    @property
    def agent_id(self) -> str:
        """
        Get the agent id.

        :return: The agent id.
        """
        return self._agent_id

    @property
    def metadata(self) -> AgentMetadata:
        """ Metadata describing the physical properties of the agent. """
        return self._metadata

    def add_state(self, state: State):
        """
        Add a state to the trajectory of the agent.

        :param state: The state to add.
        """
        self._state_trajectory.append(state)

    def add_observation(self, observation: Observation):
        """
        Add an observation to the trajectory of the agent.

        :param observation: The observation to add.
        """
        self._obs_trajectory.append(observation)

    def add_action(self, action: Action):
        """
        Add an action to the trajectory of the agent.

        :param action: The action to add.
        """
        self._action_trajectory.append(action)

    @property
    def state_trajectory(self) -> [State]:
        """ The trajectory of the agent. """
        return self._state_trajectory

    @property
    def observation_trajectory(self) -> [Observation]:
        """ The trajectory of the agent. """
        return self._obs_trajectory

    @property
    def action_trajectory(self) -> [Action]:
        """ The trajectory of the agent. """
        return self._action_trajectory

    @property
    def nearby_vehicles(self) -> [Dict[str, Any]]:
        """ The nearby vehicles at each time step. """
        return self.__evaluation_features_trajectory["nearby_vehicles"]

    def add_nearby_vehicles(self, nearby_vehicles: Dict[str, Any]):
        """
        Add the nearby vehicles to the trajectory of the agent.

        :param nearby_vehicles: The nearby vehicles to add.
        """

        # for each vehicle, get the position, velocity and metadata
        nearby_vehicles_new = {}
        for position, vehicle in nearby_vehicles.items():

            if vehicle is not None:
                vehicle = {"agent_id": vehicle.agent_id, "position": vehicle.state.position,
                           "speed": vehicle.state.speed, "metadata": vehicle.metadata}

            nearby_vehicles_new[position] = vehicle

        self.__evaluation_features_trajectory["nearby_vehicles"].append(nearby_vehicles_new)

    @property
    def distance_right_lane_marking(self):
        return self.__evaluation_features_trajectory["distance_right_lane_marking"]

    @property
    def distance_left_lane_marking(self):
        return self.__evaluation_features_trajectory["distance_left_lane_marking"]

    def add_distance_right_lane_marking(self, distance_right_lane_marking: float):
        self.__evaluation_features_trajectory["distance_right_lane_marking"].append(distance_right_lane_marking)

    def add_distance_left_lane_marking(self, distance_left_lane_marking: float):
        self.__evaluation_features_trajectory["distance_left_lane_marking"].append(distance_left_lane_marking)

    @property
    def meta(self):
        return self._metadata

    @property
    def initial_state(self) -> State:
        """ The initial state of the agent. """
        return self._initial_state

    @property
    def original_initial_time(self) -> float:
        """ The time at which the agent was created in the DATASET (i.e., not when spawned in the simulator). """
        return self._original_initial_time

    @property
    def state(self) -> State:
        """The last state in the trajectory"""
        return self.state_trajectory[-1]


class DummyRandomAgent:
    """
    An agent that is temporarily used to spawn a vehicle at a random location.
    Needs to have the same interface as an agent in the AUTOMATUM dataset.
    """

    def __init__(self, UUID: str, length: float, width: float, type: str, initial_position: np.ndarray,
                 initial_heading: float, initial_time: float, initial_speed: List, initial_acceleration: List):

        self.UUID = UUID
        self.length = length
        self.width = width
        self.type = type
        self.x_vec = [initial_position[0]]
        self.y_vec = [initial_position[1]]
        self.psi_vec = [initial_heading]
        self.vx_vec = [initial_speed[0]]
        self.vy_vec = [initial_speed[1]]
        self.ax_vec = [initial_acceleration[0]]
        self.ay_vec = [initial_acceleration[1]]
        self.time = [initial_time]



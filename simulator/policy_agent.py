"""
Class to define an agent that follows a given policy pi(s) -> (acceleration, steering_angle).
"""
from collections import defaultdict
from typing import Any, Dict
from simulator.state_action import Observation, Action, State
from sim4ad.agentstate import AgentMetadata

import logging

logger = logging.getLogger(__name__)

class PolicyAgent:
    def __init__(self, agent: Any, policy, initial_state: State):
        """
        Initialize a new PolicyAgent with the given policy.

        :param agent: The agent as loaded from the episode.
        :param policy: A function that takes history of observations and returns acceleration and steering angle.
        """

        self._agent_id = agent.UUID
        self._alive = True
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
        self._step = 0
        self._initial_state = initial_state

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
        self._step += 1

        return action

    @staticmethod
    def reached_goal(state: State) -> bool:
        """
        Check if the agent has reached the goal.

        :param state: The current state of the agent.
        :return: Whether the agent has reached the goal.
        """

        reached_end_lane = state.lane.distance_at(state.position) > 0.98 * state.lane.length
        if reached_end_lane is True:
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

    @property
    def alive(self) -> bool:
        """ Whether the agent is alive in the simulation. """
        return self._alive

    @alive.setter
    def alive(self, value: bool):
        self._alive = value

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
    def state(self) -> State:
        """The last state in the trajectory"""
        return self.state_trajectory[-1]


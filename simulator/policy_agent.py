"""
Class to define an agent that follows a given policy pi(s) -> (acceleration, steering_angle).
"""
from typing import Callable, Tuple
from simulator.state_action import Observation, Action, State
import numpy as np
from sim4ad.agentstate import AgentMetadata


class PolicyAgent:
    def __init__(self, agent_id: str, policy: Callable[[[Observation]], Tuple[float, float]],
                 initial_state: State, metadata: AgentMetadata = None):
        """
        Initialize a new PolicyAgent with the given policy.

        :param agent_id: The agent id.
        :param metadata: The metadata describing the physical properties of the agent.
        :param policy: A function that takes history of observations and returns acceleration and steering angle.
        :param initial_state: The initial state of the agent.
        """

        self._agent_id = agent_id
        self._alive = True
        self._metadata = metadata if metadata is not None else AgentMetadata(initial_time=initial_state.time,
                                                                             **AgentMetadata.CAR_DEFAULT)
        self.policy = policy
        self._trajectory = [initial_state]
        self._step = 0
        self._initial_state = initial_state
        self._step_max = 1000  # TODO: set to a reasonable value

        # For the bicycle model
        correction = (self._metadata.rear_overhang - self._metadata.front_overhang) / 2  # Correction for cg
        self._l_f = self._metadata.wheelbase / 2 + correction  # Distance of front axel from cg
        self._l_r = self._metadata.wheelbase / 2 - correction  # Distance of back axel from cg

    def next_action(self, history: [[Observation]]) -> Action:
        """
        Get the action for the given state history.

        :param history: The history of the observations/states.
        :return: The desired acceleration and steering angle. TODO: update
        """
        acceleration, delta = self.policy(history)[0].tolist()

        action = Action(acceleration=acceleration, steer_angle=delta)
        self._step += 1

        # Run the bicycle model to get the next state (position, velocity, heading)

        # Compute the new Observation based on the new Observation and Action
        # TODO: add to traj
        # TODO: change life status

        return action

    def done(self):
        # TODO: could do that is also done if reached goal location ?
        return self._step >= self._step_max

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

    @property
    def trajectory(self):
        """ The trajectory that was actually driven by the agent. """
        return self._trajectory

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
        return self.trajectory[-1]


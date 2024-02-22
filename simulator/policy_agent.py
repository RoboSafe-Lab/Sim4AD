"""
Class to define an agent that follows a given policy pi(s) -> (acceleration, steering_angle).
"""
from typing import Callable, Tuple, Any, Dict
from simulator.state_action import Observation, Action, State
import numpy as np
from sim4ad.agentstate import AgentMetadata


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
        self.__evaluation_features_trajectory = []  # List of all the features used for evaluation at each time step.
        self._step = 0
        self._initial_state = initial_state
        self._step_max = 1000  # TODO: set to a reasonable value ??? WHAT IS THIS FOR??

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
    def evaluation_features_trajectory(self) -> [Dict[str, Any]]:
        """ List of all the features used for evaluation at each time step. """
        return self.__evaluation_features_trajectory

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


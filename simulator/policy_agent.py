"""
Class to define an agent that follows a given policy pi(s) -> (acceleration, steering_angle).
"""
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
from stable_baselines3 import SAC

from baselines.idm import IDM
from simulator.simulator_util import PositionNearbyAgent
from simulator.state_action import Observation, Action, State
from sim4ad.agentstate import AgentMetadata
from baselines.bc_baseline import BCBaseline as BC
from sim4ad.common_constants import MISSING_NEARBY_AGENT_VALUE
import logging

logger = logging.getLogger(__name__)


class PolicyAgent:
    def __init__(self, agent: Any, policy, initial_state: State, original_initial_time: float, device="cpu"):
        """
        Initialize a new PolicyAgent with the given policy.

        :param agent: The agent as loaded from the episode.
        :param policy: A function that takes history of observations and returns acceleration and steering angle.

        :param original_initial_time: The time at which the corresponding agent was created in the DATASET.
        """

        self._agent_id = agent.UUID
        self._metadata = AgentMetadata(length=agent.length, width=agent.width, agent_type=agent.type,
                                       front_overhang=0.91, rear_overhang=1.094,
                                       wheelbase=agent.length*0.6,
                                       max_acceleration=1.5,
                                       max_angular_vel=2.0
                                       )

        self.policy = policy
        self._state_trajectory = [initial_state]
        self._obs_trajectory = []
        self._action_trajectory = []
        # List of all the features used for evaluation at each time step.
        self.__evaluation_features_trajectory = defaultdict(list)
        self._initial_state = initial_state
        self._original_initial_time = original_initial_time
        self.__long_acc = []
        self.__lat_acc = []
        self.__long_jerk = []
        self.idm = IDM()

        # For the bicycle model
        correction = (self._metadata.rear_overhang - self._metadata.front_overhang) / 2  # Correction for cg
        self._l_f = self._metadata.wheelbase / 2 + correction  # Distance of front axel from cg
        self._l_r = self._metadata.wheelbase / 2 - correction  # Distance of back axel from cg

        self.device = device

    def next_action(self, history: [[Observation]]) -> Action:
        """
        Get the action for the given state history.

        :param history: The history of the observations/states.
        :return: The desired acceleration and steering angle.
        """

        obs = history[-1]
        if isinstance(self.policy, SAC):
            (acceleration, delta), _ = self.policy.predict(obs, deterministic=True)
        elif isinstance(self.policy, BC):
            acceleration, delta = self.policy(history)[0].tolist()
        else:
            acceleration, delta = self.policy.act(obs, device=self.device, deterministic=True)

        action = Action(acceleration=acceleration, yaw_rate=delta)

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
        return reached_end_lane

    def __call__(self, history: [[Observation]]) -> Action:
        """
        Get the action for the given state history.

        :param history: The history of the observations/states.
        :return: The desired acceleration and steering angle.
        """
        return self.next_action(history)

    def terminated(self, max_steps: int) -> bool:
        return len(self._state_trajectory) >= max_steps

    def compute_current_lat_lon_acceleration(self):
        """
        Compute the lateral and longitudinal acceleration at the current timestep.
        :return: lat ang long acceleration
        """

        state = self.state_trajectory[-1]
        acc = state.acceleration
        long_acc = acc * np.cos(state.heading)
        lat_acc = acc * np.sin(state.heading)

        assert (np.sqrt(long_acc ** 2 + lat_acc ** 2) - np.abs(state.acceleration) < 1e-6), \
            f"Computed acceleration: {np.sqrt(long_acc ** 2 + lat_acc ** 2)}, " \
            f"state acceleration: {state.acceleration}"

        self.__long_acc.append(long_acc)
        self.__lat_acc.append(lat_acc)

        return lat_acc, long_acc

    def compute_current_long_jerk(self):
        """
        Compute the longitudinal jerk at the current timestep.
        :return: long jerk
        """

        assert len(self.__long_acc) == len(self._state_trajectory)

        long_jerk = 0
        if len(self.__long_acc) > 2:
            long_jerk = (self.__long_acc[-1] - self.__long_acc[-2]) / \
                        (self._state_trajectory[-1].time - self._state_trajectory[-2].time)

        self.__long_jerk.append(long_jerk)
        return long_jerk
    
    def check_adjacent_lanes(self):
        state = self.state_trajectory[-1]
        current_lane_id = state.lane.id
        lane_section = state.line.lane_section

        # Initialize availability of adjacent lanes
        left_lane_available = False
        right_lane_available = False

        # Search through all lanes in the lane section
        for l in lane_section.all_lanes:
            if current_lane_id > 0:
                # Right-side lanes: check for left lane (id - 1) and right lane (id + 1)
                if l.id == current_lane_id - 1 and l.id != 1:
                    left_lane_available = True
                if l.id == current_lane_id + 1 and l.id != 5:
                    right_lane_available = True
            elif current_lane_id < 0:
                # Left-side lanes: check for left lane (id + 1) and right lane (id - 1)
                if l.id == current_lane_id + 1 and l.id != -1:
                    left_lane_available = True
                if l.id == current_lane_id - 1 and l.id != -5:
                    right_lane_available = True

        return left_lane_available, right_lane_available


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

    def last_vehicle_in_front_ego(self):
        """
        Get the vehicle in front of the ego agent at the last time step.
        """
        return self.__evaluation_features_trajectory["nearby_vehicles"][-1][PositionNearbyAgent.CENTER_IN_FRONT]

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
                           "velocity": vehicle.state.velocity, "metadata": vehicle.metadata,
                           "heading": vehicle.state.heading}

            nearby_vehicles_new[position] = vehicle

        self.__evaluation_features_trajectory["nearby_vehicles"].append(nearby_vehicles_new)
        return nearby_vehicles_new

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

    def add_distance_midline(self, distance_midline: float):
        self.__evaluation_features_trajectory["distance_midline"].append(distance_midline)

    @property
    def meta(self):
        return self._metadata

    @property
    def distance_midline(self):
        return self.__evaluation_features_trajectory["distance_midline"]

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

    def add_interference(self, agent_id):
        self.__evaluation_features_trajectory["interferences"].append(agent_id)

    @property
    def interferences(self):
        return self.__evaluation_features_trajectory["interferences"]


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



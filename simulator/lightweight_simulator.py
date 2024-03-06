"""
This file is used to run a given policy on a given scenario and generate the trajectories of the vehicles in the
scenario.

The basic structure is vaguely based on https://github.com/uoe-agents/IGP2/blob/main/igp2/simplesim/simulation.py
"""
import logging
import random
from collections import defaultdict
from copy import deepcopy
from typing import Tuple, List, Dict, Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from shapely import Point

from baselines.bc_baseline import PolicyNetwork as BC
from extract_observation_action import ExtractObservationAction
from sim4ad.common_constants import MISSING_NEARBY_AGENT_VALUE
from sim4ad.data import DatasetDataLoader, Episode
from sim4ad.opendrive import plot_map, Lane, Map
from sim4ad.util import Box
from simulator.policy_agent import PolicyAgent, DummyRandomAgent
from simulator.state_action import State, Action, Observation
from simulator.simulator_util import DeathCause
from simulator.simulator_util import PositionNearbyAgent as PNA
from evaluation.evaluation_functions import EvaluationFeaturesExtractor

logger = logging.getLogger(__name__)


class Sim4ADSimulation:

    def __init__(self,
                 scenario_map: Map,
                 dt: float = 0.1,
                 open_loop: bool = False,
                 episode_agents: Dict[str, Any] = None,
                 policy_type: str = "bc",
                 simulation_name: str = "sim4ad_simulation",
                 spawn_method: str = "dataset"):

        """ Initialise new simulation.

        Args:
            scenario_map: The current road layout.
            dt: Time difference between two time steps.
            open_loop: If true then no physical controller will be applied.
            episode_agents: The agents in the episode. As a dictionary (agent_id, agent).
            policy_type: The type of policy to use.
            simulation_name: The name of the simulation.
            spawn_method: The method to spawn the vehicles in the simulation. Either "dataset" or "random".
        """
        self.__scenario_map = scenario_map
        self.__dt = dt
        self.__fps = np.round(1 / dt)
        self.__open_loop = open_loop

        self.__time = 0
        self.__state = {}
        self.__agents = {}
        self.__episode_agents = episode_agents
        self.__agents_to_add = deepcopy(self.__episode_agents)  # Agents that have not been added to the simulation yet.
        self.__simulation_history = []  # History of frames (agent_id, State) of the simulation.
        self.__simulation_name = simulation_name
        self.__eval = EvaluationFeaturesExtractor(sim_name=simulation_name)
        self.__last_agent_id = 0

        # dataset_all: agents are spawned at the time they appear in the dataset, but are controlled by the policy.
        # random: agents are spawned at random times and positions.
        # dataset-one: all but one agent follow the dataset, the other one, is controlled by the policy.
        assert spawn_method in ["dataset_all", "random", "dataset_one"], f"Spawn method {spawn_method} not found."

        self.__spawn_method = spawn_method

        assert policy_type == "follow_dataset" or "bc" in policy_type.lower(), f"Policy type {policy_type} not found."
        if policy_type == "follow_dataset":
            assert spawn_method != "random", "Policy type 'follow_dataset' is not compatible with 'random' spawn"

        self.__policy_type = policy_type

        # Dictionary (agent_id, DeathCause) of agents that have been removed from the simulation.
        self.__dead_agents = {}
        self.__agent_evaluated = None  # If we spawn_method is "dataset_one", then this is the agent we are evaluating.

    def add_agent(self, new_agent: PolicyAgent):
        """ Add a new agent to the simulation.

        Args:
            new_agent: Agent to add.
        """
        if new_agent.agent_id in self.__agents \
                and self.__agents[new_agent.agent_id] is not None:
            raise ValueError(f"Agent with ID {new_agent.agent_id} already exists.")

        self.__agents[new_agent.agent_id] = new_agent

        if self.__spawn_method in ["dataset_all", "random"] or new_agent.agent_id == self.__agent_evaluated:
            self.__state[new_agent.agent_id] = new_agent.initial_state
        elif self.__spawn_method == "dataset_one":
            # Spawn the vehicle at the state it was in the dataset at the time of the simulation.
            # TODO: maybe make a method to create a policy agent from dataset as we use it in multiple places.
            dataset_time_step = round((self.__time - new_agent.initial_state.time) / self.__dt)
            new_agent_original = self.__episode_agents[new_agent.agent_id]
            position = Point(new_agent_original.x_vec[dataset_time_step], new_agent_original.y_vec[dataset_time_step])
            speed = np.sqrt(float(new_agent_original.vx_vec[dataset_time_step]) ** 2 + float(new_agent_original.vy_vec[dataset_time_step]) ** 2)
            heading = new_agent_original.psi_vec[dataset_time_step]
            lane = self.__scenario_map.best_lane_at(position, heading)
            current_state = State(time=new_agent_original.time[dataset_time_step], position=position, speed=speed,
                                  acceleration=np.sqrt(float(new_agent_original.ax_vec[dataset_time_step]) ** 2 +
                                                       float(new_agent_original.ay_vec[dataset_time_step]) ** 2),
                                  heading=heading, lane=lane, agent_width=new_agent_original.width,
                                  agent_length=new_agent_original.length)

            # Add it to the current state
            self.__state[new_agent.agent_id] = current_state
        else:
            raise ValueError(f"Spawn method {self.__spawn_method} not found.")

        if self.__spawn_method == "dataset_all":
            self.__agents_to_add.pop(new_agent.agent_id)
        elif self.__spawn_method == "dataset_one" and new_agent.agent_id == self.__agent_evaluated:
            self.__agents_to_add.pop(new_agent.agent_id)
        elif self.__spawn_method == "random":
            # There is no list of agents to add.
            pass

        logger.debug(f"Added Agent {new_agent.agent_id}")

    @staticmethod
    def _get_bc_policy(baseline_name: str = "bc-1epoch"):
        """
        Load the BC policy.
        """
        # TODO: we predict (acceleration, steering angle) from the history of observations
        policy = BC()
        policy.load_policy(baseline_name=baseline_name)
        return policy

    def _create_policy_agent(self, agent, policy: str = "bc"):

        center = np.array([float(agent.x_vec[0]), float(agent.y_vec[0])])
        heading = agent.psi_vec[0]
        lane = self.__scenario_map.best_lane_at(center, heading)
        initial_state = State(time=agent.time[0], position=center,
                              speed=np.sqrt(float(agent.vx_vec[0]) ** 2 + float(agent.vy_vec[0]) ** 2),
                              acceleration=np.sqrt(float(agent.ax_vec[0]) ** 2 + float(agent.ay_vec[0]) ** 2),
                              heading=heading, lane=lane, agent_width=agent.width, agent_length=agent.length)

        if "bc" in policy.lower():
            policy = self._get_bc_policy(baseline_name=policy)
        elif policy == "follow_dataset":
            pass
        else:
            raise ValueError(f"Policy {policy} not found.")

        return PolicyAgent(agent=agent, policy=policy, initial_state=initial_state)

    def remove_agent(self, agent_id: str, death_cause: DeathCause):
        """ Remove an agent from the simulation.

        Args:
            agent_id: Agent ID to remove.
        """

        agent_removed = self.__agents.pop(agent_id)

        # Save the agent's trajectory for evaluation
        if self.__spawn_method in ["dataset_all", "random"] or agent_id == self.__agent_evaluated:
            self.__eval.save_trajectory(agent=agent_removed, death_cause=death_cause)

        logger.debug(f"Removed Agent {agent_id}")

    def kill_all_agents(self):
        """ Remove all agents from the simulation with TIMEOUT death. """

        for agent_id in list(self.__agents):
            self.remove_agent(agent_id, DeathCause.TIMEOUT)

        self.__agents = {}

    def reset(self):
        """ Remove all agents and reset internal state of simulation. """
        self.__time = 0
        self.__agents = {}
        self.__state = {}
        self.__dead_agents = {}
        self.__agents_to_add = deepcopy(self.__episode_agents)
        self.__simulation_history = []
        raise NotImplementedError("Check there is nothing else to reset.")  # TODO

    def step(self):
        """ Advance simulation by one time step. """
        logger.debug(f"Simulation time {self.__time + self.__dt}")
        self.__update_vehicles()
        self.__take_actions()
        self.__time += self.__dt

    @property
    def time(self):
        """ Get the current time of the simulation. """
        return self.__time

    def __update_vehicles(self):
        """
        Spawn new vehicles in the simulation based on when they appear in the dataset.
        Remove the vehicles that are dead.
        """

        # REMOVE DEAD AGENTS
        for agent_id, death_cause in self.__dead_agents.items():
            self.remove_agent(agent_id, death_cause)
            logger.debug(f"Agent {agent_id} has been removed from the simulation for {death_cause} at t={self.time}.")

        # If we removed the agent we are evaluating, and we are in "dataset_one" mode, then we need to remove all.
        if self.__spawn_method == "dataset_one" and self.__agent_evaluated in self.__dead_agents:
            self.__agents = {}
            self.__agent_evaluated = None

        self.__dead_agents = {}

        # SELECT AGENTS TO ADD
        add_agents = {}

        if self.__spawn_method == "dataset_all":
            for agent_id, agent in self.__agents_to_add.items():
                if agent.time[0] <= self.__time <= agent.time[-1]:
                    if agent_id not in self.__agents:
                        add_agents[agent_id] = (agent, self.__policy_type)
        elif self.__spawn_method == "random":
            if self.__time == 0:
                self.__spawn_features, self.__possible_vehicle_dimensions = self._get_spawn_positions()

            spawn_probability = 0.05  # TODO: make it a parameter
            if random.random() < spawn_probability:
                agent_to_spawn = self._get_vehicle_to_spawn()
                add_agents[agent_to_spawn.UUID] = (agent_to_spawn, self.__policy_type)

        elif self.__spawn_method == "dataset_one":
            # We want to iterate through the dataset, and add one agent at a time that follows our policy. Then, spawn
            # those that are at the same time as the dataset as following the trajectory. Then, when the agent is dead,
            # remove it and add the next one, starting the time of the simulator back to the spawn time of the agent.

            if self.__agent_evaluated is None:
                # Get the first agent in the dataset
                self.__agent_evaluated = list(self.__agents_to_add.keys())[0]
                self.__time = self.__agents_to_add[self.__agent_evaluated].time[0]

            # Spawn all vehicles alive at the current time.
            for agent_id, agent in self.__episode_agents.items():
                if agent.time[0] <= self.__time <= agent.time[-1]:
                    if agent_id not in self.__agents:
                        if agent_id == self.__agent_evaluated:
                            add_agents[agent_id] = (agent, self.__policy_type)
                        else:
                            add_agents[agent_id] = (agent, "follow_dataset")
        else:
            raise ValueError(f"Spawn method {self.__spawn_method} not found.")

        # ADD AGENTS
        for agent_id, (agent, policy) in add_agents.items():
            self.add_agent(self._create_policy_agent(agent, policy=policy))

    def _get_vehicle_to_spawn(self):

        for _ in range(1000):  # Try 1000 times to spawn a vehicle, otherwise fail
            # Sample a random position and dimension, then check if we can spawn a vehicle there (i.e., if it is not
            # already occupied by another vehicle)
            spawn_lane = random.choice(list(self.__spawn_features.keys()))
            spawn_distance = random.choice(self.__spawn_features[spawn_lane]["distance"])
            spawn_vx = random.choice(self.__spawn_features[spawn_lane]["vx"])
            spawn_vy = random.choice(self.__spawn_features[spawn_lane]["vy"])
            spawn_heading = random.choice(self.__spawn_features[spawn_lane]["heading"])
            spawn_ax = random.choice(self.__spawn_features[spawn_lane]["ax"])
            spawn_ay = random.choice(self.__spawn_features[spawn_lane]["ay"])

            agent_type, spawn_width, spawn_length = random.choice(self.__possible_vehicle_dimensions)

            spawn_position = spawn_lane.point_at(spawn_distance)

            # Check if the position is free
            bbox = Box(center=spawn_position, length=spawn_length, width=spawn_width, heading=spawn_heading)

            safe_to_spawn = True
            for agent in self.__agents.values():
                if agent.state.bbox.overlaps(bbox):
                    safe_to_spawn = False
                    break

            if safe_to_spawn is False:
                continue

            agent_id = f"random_agent_{self.__last_agent_id}"
            self.__last_agent_id += 1

            agent = DummyRandomAgent(UUID=agent_id, length=spawn_length, width=spawn_width, type=agent_type,
                                     initial_position=spawn_position,
                                     initial_heading=spawn_heading, initial_time=self.__time,
                                     initial_speed=[spawn_vx, spawn_vy],
                                     initial_acceleration=[spawn_ax, spawn_ay])
            return agent

        raise ValueError("Could not spawn a vehicle after 1000 attempts.")

    def _get_spawn_positions(self):
        """
        Get from the dataset the initial positions, headings, and speeds of the agents.

        :return: List of spawn positions.
        """

        # For each lane, store the distances from the start of the lane where the agents were spawned in the dataset,
        # the speeds, and the headings. Key: lane, Value: (distance, speed, heading, ...)
        possible_lanes = defaultdict(lambda: {"distance": [], "vx": [], "vy": [], "heading": [], "ax": [], "ay": []})
        vehicle_dimensions = []  # store it as a list of tuples (width, length)

        for _, agent in self.__episode_agents.items():
            # Get the distances from the start of the lane to the start of the road

            initial_position = Point(agent.x_vec[0], agent.y_vec[0])
            initial_heading = agent.psi_vec[0]
            lane = self.__scenario_map.best_lane_at(initial_position, initial_heading)
            distance = lane.distance_at(initial_position)  # where the agent was spawned

            # Store the distance, speed, and heading
            possible_lanes[lane]["distance"].append(distance)
            possible_lanes[lane]["vx"].append(agent.vx_vec[0])
            possible_lanes[lane]["vy"].append(agent.vy_vec[0])
            possible_lanes[lane]["heading"].append(initial_heading)
            possible_lanes[lane]["ax"].append(agent.ax_vec[0])
            possible_lanes[lane]["ay"].append(agent.ay_vec[0])

            vehicle_dimensions.append((agent.type, agent.width, agent.length))

        return possible_lanes, vehicle_dimensions

    def __take_actions(self):

        # A frame is a dictionary (agent_id, State)
        new_frame = {}

        # Compute the observation for each agent for the current state
        for agent_id, agent in self.__agents.items():

            if agent is None:
                continue

            # TODO: for efficiency we could compute the observations of all agents at once
            obs, vehicles_nearby = self._get_observation(agent=agent, state=self.__state[agent_id])

            # Compute the evaluation features for the agent
            evaluation = True  # TODO: make it parameter
            if evaluation is True:
                agent.add_nearby_vehicles(vehicles_nearby)
                agent.add_distance_right_lane_marking(obs.get_feature("distance_right_lane_marking"))
                agent.add_distance_left_lane_marking(obs.get_feature("distance_left_lane_marking"))

            collision = self.__collision(agent_state=self.__state[agent_id], nearby_vehicles=vehicles_nearby)

            if collision is True:
                self.__dead_agents[agent_id] = DeathCause.COLLISION

            # Put the observation in a tuple, as the policy expects it
            obs = obs.get_tuple()
            agent.add_observation(obs)

        # Given the above observations, choose an action and compute the next state
        for agent_id, agent in self.__agents.items():
            if agent is None and agent_id not in self.__dead_agents: # Check if agent e.g., if collision
                continue

            if agent.policy == "follow_dataset":
                dataset_time_step = round((self.__time - agent.initial_state.time) / self.__dt)
                # Get the acceleration and steering angle from the dataset
                dataset_agent = self.__episode_agents[agent_id]
                deltas = ExtractObservationAction.extract_steering_angle(agent=dataset_agent)
                acceleration = np.sqrt(float(dataset_agent.ax_vec[dataset_time_step]) ** 2 +
                                       float(dataset_agent.ay_vec[dataset_time_step]) ** 2)
                action = Action(acceleration=acceleration,
                                steer_angle=deltas[dataset_time_step])
            else:
                action = agent.next_action(history=agent.observation_trajectory)

            agent.add_action(action)
            # Use the bicycle model to find where the agent will be at t+1
            new_state = self._next_state(agent, current_state=self.__state[agent_id], action=action)

            done = False
            if agent.policy == "follow_dataset":

                if dataset_time_step+1 < len(dataset_agent.x_vec):
                    position = Point(dataset_agent.x_vec[dataset_time_step + 1],
                                            dataset_agent.y_vec[dataset_time_step + 1])
                    speed = np.sqrt(float(dataset_agent.vx_vec[dataset_time_step + 1]) ** 2 +
                                    float(dataset_agent.vy_vec[dataset_time_step + 1]) ** 2)
                    heading = dataset_agent.psi_vec[dataset_time_step + 1]
                    acceleration = np.sqrt(float(dataset_agent.ax_vec[dataset_time_step + 1]) ** 2 +
                                           float(dataset_agent.ay_vec[dataset_time_step + 1]) ** 2)
                    new_state = State(time=new_state.time, position=position, speed=speed, acceleration=acceleration,
                                      heading=heading, lane=new_state.lane, agent_width=agent.meta.width,
                                      agent_length=agent.meta.length)
                else:
                    # The agent has reached the end of the dataset
                    done = True
            else:
                done = agent.reached_goal(new_state)

            off_road = new_state.lane is None

            if done is True:
                self.__dead_agents[agent_id] = DeathCause.GOAL_REACHED  # todo: what if goal reached?
            elif off_road is True:
                self.__dead_agents[agent_id] = DeathCause.OFF_ROAD

            new_frame[agent_id] = new_state
            agent.add_state(new_state)  # TODO: agent.trajectory.add_state(new_state, reload_path=False)

        new_frame["time"] = self.__time + self.__dt
        self.__simulation_history.append(new_frame)
        self.__state = new_frame


    def _next_state(self, agent: PolicyAgent, current_state: State, action: Action) -> State:
        """
        Compute the next state based on the current state and action using the bicycle model.

        Apply acceleration and steering according to the bicycle model centered at the center-of-gravity (i.e. cg)
        of the vehicle.

        Ref: https://dingyan89.medium.com/simple-understanding-of-kinematic-bicycle-model-81cac6420357

        :param current_state: The current State.
        :param action: Acceleration and steering action to execute
        :return: The next State.
        """

        acceleration = np.clip(action.acceleration, - agent.meta.max_acceleration, agent.meta.max_acceleration)

        speed = current_state.speed + acceleration * self.__dt
        speed = max(0, speed)

        beta = np.arctan(agent._l_r * np.tan(action.steer_angle) / agent.meta.wheelbase)
        d_position = np.array(
            [speed * np.cos(beta + current_state.heading),
             speed * np.sin(beta + current_state.heading)]
        )

        center = np.array([current_state.position.x, current_state.position.y]) + d_position * self.__dt
        d_theta = speed * np.tan(action.steer_angle) * np.cos(beta) / agent.meta.wheelbase
        d_theta = np.clip(d_theta, - agent.meta.max_angular_vel, agent.meta.max_angular_vel)
        heading = (current_state.heading + d_theta * self.__dt + np.pi) % (2 * np.pi) - np.pi

        new_lane = self.__scenario_map.best_lane_at(center, heading)

        # TODO: is the time correct? or should we use the time of the action?
        return State(time=self.time + self.dt, position=center, speed=speed, acceleration=acceleration,
                     heading=heading, lane=new_lane, agent_width=agent.meta.width, agent_length=agent.meta.length)

    def _get_observation(self, agent: PolicyAgent, state: State) -> Tuple[Observation, dict]:
        """
        Get the current observation of the agent.

        :param agent: The agent.
        :return: The observation and the nearby agents.
        """

        distance_left_lane_marking, distance_right_lane_marking = self._compute_distance_markings(state=state)

        # nearby_agents_features contains dx, dy, v, a, heading for each nearby agent
        # vehicles_nearby contains the agent object
        nearby_agents_features, vehicles_nearby = self._get_nearby_vehicles(agent=agent, state=state)

        front_ego = nearby_agents_features[PNA.CENTER_IN_FRONT]
        behind_ego = nearby_agents_features[PNA.CENTER_BEHIND]
        left_front = nearby_agents_features[PNA.LEFT_IN_FRONT]
        left_behind = nearby_agents_features[PNA.LEFT_BEHIND]
        right_front = nearby_agents_features[PNA.RIGHT_IN_FRONT]
        right_behind = nearby_agents_features[PNA.RIGHT_BEHIND]

        observation = {
            "speed": state.speed,
            "heading": state.heading,
            "distance_left_lane_marking": distance_left_lane_marking,
            "distance_right_lane_marking": distance_right_lane_marking,
            "x": state.position.x, # TODO: remove x and y
            "y": state.position.y,
            "front_ego_rel_dx": front_ego["rel_dx"],
            "front_ego_rel_dy": front_ego["rel_dy"],
            "front_ego_rel_speed": front_ego["speed"] - state.speed,
            "front_ego_rel_a": front_ego["a"] - state.acceleration,
            "front_ego_heading": front_ego["heading"],
            "behind_ego_rel_dx": behind_ego["rel_dx"],
            "behind_ego_rel_dy": behind_ego["rel_dy"],
            "behind_ego_rel_speed": behind_ego["speed"] - state.speed,
            "behind_ego_rel_a": behind_ego["a"] - state.acceleration,
            "behind_ego_heading": behind_ego["heading"],
            "front_left_rel_dx": left_front["rel_dx"],
            "front_left_rel_dy": left_front["rel_dy"],
            "front_left_rel_speed": left_front["speed"] - state.speed,
            "front_left_rel_a": left_front["a"] - state.acceleration,
            "front_left_heading": left_front["heading"],
            "behind_left_rel_dx": left_behind["rel_dx"],
            "behind_left_rel_dy": left_behind["rel_dy"],
            "behind_left_rel_speed": left_behind["speed"] - state.speed,
            "behind_left_rel_a": left_behind["a"] - state.acceleration,
            "behind_left_heading": left_behind["heading"],
            "front_right_rel_dx": right_front["rel_dx"],
            "front_right_rel_dy": right_front["rel_dy"],
            "front_right_rel_speed": right_front["speed"] - state.speed,
            "front_right_rel_a": right_front["a"] - state.acceleration,
            "front_right_heading": right_front["heading"],
            "behind_right_rel_dx": right_behind["rel_dx"],
            "behind_right_rel_dy": right_behind["rel_dy"],
            "behind_right_rel_speed": right_behind["speed"] - state.speed,
            "behind_right_rel_a": right_behind["a"] - state.acceleration,
            "behind_right_heading": right_behind["heading"]
        }

        observation = Observation(state=observation)
        return observation, vehicles_nearby

    def _compute_distance_markings(self, state: State) -> Tuple[float, float]:
        """
        Compute the distance to the left and right lane markings.

        :param previous_state: The previous state.
        :param state: The current state.
        :return: The distance to the left and right lane markings.
        """

        lane = state.lane

        if lane is None:
            raise ValueError(f"No lane found at position {state.position} and heading {state.heading}.")

        ds_on_lane = lane.distance_at(state.position)

        # 1. find the slope of the line perpendicular to the lane through the agent

        position = state.position
        # Find the point on the boundary closest to the agent
        closest_point = lane.boundary.boundary.interpolate(lane.boundary.boundary.project(position))

        # We now want to find if the point is on the left or right side of the agent
        # We can do this by using the cross product of the vector v1 from (init_x,init_y,0) to the agent and
        # vector v2 from (init_x,init_y,0) to the closest point, where init_x, and init_y is the first point in the
        # of the midline of the lane. If the cross product is negative, then the point is on the right side of the agent,
        # if it is positive, then it is on the left side.
        v_init = lane.point_at(0)
        v_init = np.array([v_init[0], v_init[1], 0])

        v1 = np.array([state.position.x, state.position.y, 0]) - v_init
        v2 = np.array([closest_point.x, closest_point.y, 0]) - v_init
        cross_product = np.cross(v1, v2)
        if cross_product[2] < 0:  # Check the z-component of the cross product
            # The point is on the right side of the agent=
            distance_right_lane_marking = position.distance(closest_point)
            distance_left_lane_marking = lane.get_width_at(ds_on_lane) - distance_right_lane_marking
        else:
            # The point is on the left side of the agent
            distance_left_lane_marking = position.distance(closest_point)
            distance_right_lane_marking = lane.get_width_at(ds_on_lane) - distance_left_lane_marking

        assert distance_left_lane_marking + distance_right_lane_marking - lane.get_width_at(ds_on_lane) < 1e-6

        return distance_left_lane_marking, distance_right_lane_marking

    def _find_perpendicular(self, lane: Lane, state: State, length=50) -> Tuple[Point, Point]: # TODO: complete signature

        # We need to take the tangent as we want the slope (ration dy/dx) and not the heading
        ds_on_lane = lane.distance_at(state.position)
        m = -1 / np.tan(lane.get_heading_at(ds_on_lane))

        # 2. find the equation of the line to the lane through the agent: y = m * (x - x0) + y0
        y = lambda x: m * (x - state.position.x) + state.position.y

        # 3. find the points lane's width away from the agent on the left and right using
        # https://math.stackexchange.com/questions/175896/finding-a-point-along-a-line-a-certain-distance-away-from-another-point

        if state.heading < 0:
            x_left = state.position.x + length / np.sqrt(1 + m ** 2)
            x_right = state.position.x - length / np.sqrt(1 + m ** 2)
        else:
            x_right = state.position.x + length / np.sqrt(1 + m ** 2)
            x_left = state.position.x - length / np.sqrt(1 + m ** 2)

        y_left = y(x_left)
        y_right = y(x_right)

        # Check if the slope is correct
        assert (y_right - y_left) / (x_right - x_left) - m < 1e-6

        # return A and B, the left and right points.
        return Point(x_left, y_left), Point(x_right, y_right)

    def _get_nearby_vehicles(self, agent: PolicyAgent, state: State):

        """
        TODO: we assume that there is only one lane, and not consider that vehicle may be in different lane groups,
        e.g., if lane changes group in front and the agent in front is in that lane instead.
        """

        # 1. We need the id of the lane where the vehicle is on
        lane = state.lane

        if lane is None:
            raise ValueError(f"No lane found at position {state.position} and heading {state.heading}.")

        # 2. We want the lanes to the left and right of the current one (as long as they have the same flow of motion)
        # TODO: in urban environments, is this an acceptable limitation, or should we also include vehicles from
        #   the other direction, as they may surpass, merge / cut in front of the vehicle?

        nearby_lanes = {k: None for k in ["left", "center", "right"]}

        nearby_lanes["left"], nearby_lanes["center"], nearby_lanes["right"] = lane.traversable_neighbours(return_lfr_order=True)

        # 3. We want to further divide the lanes into two parts, the one in front and the one behind the vehicle.
        # We will use the perpendicular line to the lane to divide the lanes into two parts.
        # perpendicular = np.array(left_point, right_point) TODO: check this is still updated
        perpendicular = self._find_perpendicular(lane, state)

        nearby_vehicles_features = defaultdict(None)
        vehicles_nearby = defaultdict(None)

        for lane_position, nearby_lane in nearby_lanes.items():

            closest_vehicle_front = None
            closest_vehicle_behind = None

            min_distance_front = float("inf")
            min_distance_behind = float("inf")

            # Loop through all agents and check if they are in the lane
            if nearby_lane is not None:
                for nearby_agent_id, nearby_agent in self.__agents.items():
                    if nearby_agent_id == agent.agent_id:
                        continue

                    nearby_agent_position = nearby_agent.state.position

                    if nearby_lane.boundary.contains(nearby_agent_position):
                        # We now need to compute the relative position of the vehicle, whether it is in front or behind
                        # the agent, by computing the cross product of the AB vector and AP vector, where A-B are the
                        # left and right points of the perpendicular line, and P is the position of the nearby vehicle.
                        # If the cross product is positive, then the vehicle is in front of the agent, if is negative,
                        # then it is behind the agent.
                        AB = np.array([perpendicular[1].x - perpendicular[0].x, perpendicular[1].y - perpendicular[0].y, 0])
                        AP = np.array([nearby_agent_position.x - perpendicular[0].x, nearby_agent_position.y - perpendicular[0].y, 0])
                        cross_product = np.cross(AB, AP)

                        distance = state.position.distance(nearby_agent_position)

                        if cross_product[2] > 0:
                            if distance < min_distance_front:
                                min_distance_front = distance
                                closest_vehicle_front = nearby_agent
                        else:
                            if distance < min_distance_behind:
                                min_distance_behind = distance
                                closest_vehicle_behind = nearby_agent

            # For each lane, we want to store the closest vehicle in front and behind the agent
            if lane_position == "left":
                front = PNA.LEFT_IN_FRONT
                behind = PNA.LEFT_BEHIND
            elif lane_position == "center":
                front = PNA.CENTER_IN_FRONT
                behind = PNA.CENTER_BEHIND
            elif lane_position == "right":
                front = PNA.RIGHT_IN_FRONT
                behind = PNA.RIGHT_BEHIND

            nearby_vehicles_features[front] = self.__get_vehicle_features(closest_vehicle_front, state)
            nearby_vehicles_features[behind] = self.__get_vehicle_features(closest_vehicle_behind, state)

            vehicles_nearby[front] = closest_vehicle_front
            vehicles_nearby[behind] = closest_vehicle_behind

        return nearby_vehicles_features, vehicles_nearby

    @staticmethod
    def __collision(agent_state: State, nearby_vehicles: dict):
        """
        Check if the agent is colliding with a nearby vehicle.

        :param agent_state: The state of the agent.
        :param nearby_vehicles: Nearby vehicles.
        :return: True if the agent is colliding with any of the nearby vehicle, False otherwise.
        """

        for nearby_vehicle in nearby_vehicles.values():
            if nearby_vehicle is not None:
                if agent_state.bbox.overlaps(nearby_vehicle.state.bbox):
                    return True
        return False

    @staticmethod
    def __get_vehicle_features(nearby_vehicle: PolicyAgent, state: State):
        """
        :param nearby_vehicle: The vehicle to get the features from, w.r.t the ego agent.
        :param state: The current state of the ego agent.
        """
        features = defaultdict(lambda: MISSING_NEARBY_AGENT_VALUE)

        if nearby_vehicle is not None:
            features["rel_dx"] = nearby_vehicle.state.position.x - state.position.x
            features["rel_dy"] = nearby_vehicle.state.position.y - state.position.y
            features["speed"] = nearby_vehicle.state.speed
            features["a"] = nearby_vehicle.state.acceleration
            features["heading"] = nearby_vehicle.state.heading
        return features

    def replay_simulation(self):
        """
        Replay the simulation as a video using self.__simulation_history which is a list of frames, where each
        frame is a dictionary (agent_id, State). Also plot the map as background.
        """

        fig, ax = plt.subplots()

        def update(frame):
            ax.clear()
            plot_map(self.__scenario_map, markings=True, hide_road_bounds_in_junction=True, ax=ax)
            time = frame['time']
            for idx, (agent_id, state) in enumerate(frame.items()):
                if agent_id == "time":
                    continue

                # pick a color based on the hash of the agent_id
                random.seed(agent_id)
                color = (random.random(), random.random(), random.random())

                # Plot the ground truth position with a cross.
                if self.__spawn_method in ["dataset_all", "dataset_one"]:
                    original_agent = self.__episode_agents[agent_id]
                    initial_time = original_agent.time[0]
                    time_idx = int((time - initial_time) / self.__dt)
                    ax.plot(original_agent.x_vec[time_idx], original_agent.y_vec[time_idx], marker=",", color=color)

                # TODO: could use the initial=l_time above to get the start position for plotting the trajectory
                position = np.array([state.position.x, state.position.y])
                ax.plot(*position, marker=".", color=color)

                # Plot the bounding box of the agent
                bbox = state.bbox.boundary
                # repeat the first point to create a 'closed loop'
                bbox = [*bbox, bbox[0]]
                ax.plot([point[0] for point in bbox], [point[1] for point in bbox], color=color)

            ax.set_title(f"Simulation Time: {time}")

        ani = FuncAnimation(fig, update, frames=self.__simulation_history, repeat=True, interval=self.__fps) # TODO: interval=self.__fps
        plt.show()

    @property
    def dt(self):
        return self.__dt

    @property
    def evaluator(self):
        return self.__eval

    @property
    def simulation_history(self):
        return self.__simulation_history # todo: remove this


if __name__ == "__main__":

    # TODO: dynamic way of loading the map and dataset below
    scenario_map = Map.parse_from_opendrive(
        "scenarios/data/automatum/hw-a9-appershofen-001-d8087340-8287-46b6-9612-869b09e68448/staticWorld.xodr")

    # map = Map.parse_from_opendrive("scenarios/data/automatum/hw-a9-brunn-002-30250d63-e5d7-44b4-9e56-4a29534a9b09/staticWorld.xodr")

    data_loader = DatasetDataLoader(f"scenarios/configs/appershofen.json")
    data_loader.load()

    episodes = data_loader.scenario.episodes

    # TODO: loop over episodes
    episode = episodes[0]

    agent = list(episode.agents.values())[0]

    # Take the time difference between steps to be the gap in the dataset
    dt = agent.time[1] - agent.time[
        0]  # TODO: dataset.dynWorld https://openautomatumdronedata.readthedocs.io/en/latest/readme_include.html

    sim = Sim4ADSimulation(scenario_map, episode_agents=episode.agents, dt=dt, spawn_method="random",
                           policy_type="bc")
    # sim.reset()

    simulation_length = 50  # seconds

    # TODO: use tqdm
    for _ in range(int(np.floor(simulation_length / dt))):
        sim.step()

    # Remove all agents left in the simulation.
    sim.kill_all_agents()

    sim.replay_simulation()

    print("Simulation done!")

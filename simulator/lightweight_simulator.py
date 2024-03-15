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
from openautomatumdronedata.dataset import droneDataset
from shapely import Point

from baselines.bc_baseline import BCBaseline as BC
from extract_observation_action import ExtractObservationAction
from sim4ad.data import DatasetDataLoader, ScenarioConfig, DatasetScenario
from sim4ad.opendrive import plot_map, Map
from sim4ad.path_utils import get_path_to_automatum_scenario, get_path_to_automatum_map, get_config_path
from sim4ad.util import Box
from simulator.policy_agent import PolicyAgent, DummyRandomAgent
from simulator.state_action import State, Action, Observation
from simulator.simulator_util import DeathCause, get_nearby_vehicles, compute_distance_markings, collision_check
from simulator.simulator_util import PositionNearbyAgent as PNA
from evaluation.evaluation_functions import EvaluationFeaturesExtractor

logger = logging.getLogger(__name__)


class Sim4ADSimulation:

    def __init__(self,
                 scenario_map: Map,
                 dt: float = 0.1,
                 episode_agents: Dict[str, Any] = None,
                 policy_type: str = "bc",
                 simulation_name: str = "sim4ad_simulation",
                 spawn_method: str = "dataset",
                 evaluation: bool = True):

        """ Initialise new simulation.

        Args:
            scenario_map: The current road layout.
            dt: Time difference between two time steps.
            episode_agents: The agents in the episode. As a dictionary (agent_id, agent).
            policy_type: The type of policy to use.
            simulation_name: The name of the simulation.
            spawn_method: The method to spawn the vehicles in the simulation. Either "dataset_all", "dataset_one" or "random".
        """
        self.__scenario_map = scenario_map
        self.__dt = dt
        self.__fps = np.round(1 / dt)

        self.__time = 0
        self.__state = {}
        self.__agents = {}
        self.__episode_agents = episode_agents
        self.__agents_to_add = deepcopy(self.__episode_agents)  # Agents that have not been added to the simulation yet.
        self.__simulation_history = []  # History of frames (agent_id, State) of the simulation.
        self.__simulation_name = simulation_name
        self.__eval = EvaluationFeaturesExtractor(sim_name=simulation_name)
        self.__last_agent_id = 0  # ID when creating a new random agent
        self.evaluation = evaluation
        self.SPAWN_PROBABILITY = 0.02

        # dataset_all: agents are spawned at the time they appear in the dataset, but are controlled by the policy.
        # random: agents are spawned at random times and positions.
        # dataset-one: all but one agent follow the dataset, the other one, is controlled by the policy.
        assert spawn_method in ["dataset_all", "random", "dataset_one"], f"Spawn method {spawn_method} not found."

        self.__spawn_method = spawn_method

        assert policy_type in ["follow_dataset", "rl"] or "bc" in policy_type.lower(), \
            f"Policy type {policy_type} not found."
        if policy_type == "follow_dataset":
            assert spawn_method != "random", "Policy type 'follow_dataset' is not compatible with 'random' spawn"

        self.__policy_type = policy_type

        # Dictionary (agent_id, DeathCause) of agents that have been removed from the simulation.
        self.__dead_agents = {}
        self.__agent_evaluated = None  # If we spawn_method is "dataset_one", then this is the agent we are evaluating.

    def _add_agent(self, new_agent: PolicyAgent):
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
            new_agent_original = self.__episode_agents[new_agent.agent_id]
            current_state = self.__get_original_current_state(new_agent_original)

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
    def _get_policy(policy):

        if "bc" in policy.lower():  # TODO: could move ths in get_policy()
            return BC(name=policy, evaluation=True)
        elif policy == "follow_dataset":
            return "follow_dataset"
        elif policy == "rl":
            return "rl"
        else:
            raise ValueError(f"Policy {policy} not found.")

    def _create_policy_agent(self, agent, policy: str):

        original_initial_time = agent.time[0]
        initial_state = self.__get_original_current_state(agent)
        policy = self._get_policy(policy)

        return PolicyAgent(agent=agent, policy=policy, initial_state=initial_state,
                           original_initial_time=original_initial_time)

    def __get_original_current_state(self, agent):
        """
        Get the state corresponding to the agent in the dataset at the current time of the simulation.
        """

        if isinstance(agent, PolicyAgent):
            initial_time = agent.original_initial_time
        else:
            initial_time = agent.time[0]

        time_step = round((self.__time - initial_time) / self.__dt)

        center = np.array([float(agent.x_vec[time_step]), float(agent.y_vec[time_step])])
        heading = agent.psi_vec[time_step]
        lane = self.__scenario_map.best_lane_at(center, heading)

        return State(time=self.__time, position=center,
                     speed=np.sqrt(float(agent.vx_vec[time_step]) ** 2 + float(agent.vy_vec[time_step]) ** 2),
                     acceleration=np.sqrt(float(agent.ax_vec[time_step]) ** 2 + float(agent.ay_vec[time_step]) ** 2),
                     heading=heading, lane=lane, agent_width=agent.width, agent_length=agent.length)

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

    def full_reset(self):
        """ Remove all agents and reset internal state of simulation. """
        self.__time = 0
        self.__state = {}
        self.__agents = {}
        self.__agents_to_add = deepcopy(self.__episode_agents)
        self.__simulation_history = []
        self.__eval = EvaluationFeaturesExtractor(sim_name=self.__simulation_name)
        self.__last_agent_id = 0

        # Dictionary (agent_id, DeathCause) of agents that have been removed from the simulation.
        self.__dead_agents = {}
        self.__agent_evaluated = None

    def soft_reset(self):
        """
        Reset the simulation for the gym env. It means that we change the agent we are evaluating.

        :return initial_obs: The initial observation for the agent being evaluated.
                info: None
        """

        self.__simulation_history = []
        initial_obs = self.__update_vehicles(soft_reset=True)

        if initial_obs is None:
            raise ValueError("We ran out of episodes to simulate! We need to change episode/reuse previous one.")

        return initial_obs, {}

    def step(self, action: Tuple[float, float] = None):
        """
        Advance simulation by one time step.
        If action is provided, we will use that action for the agent being evaluated (only if policy is "rl").
        Order is (acceleration, steering angle).
        """

        logger.debug(f"Simulation time {self.__time}")
        self.__update_vehicles()
        ego_obs = self.__take_actions(ego_action=action)
        self.__time += self.__dt
        return ego_obs

    @property
    def time(self):
        """ Get the current time of the simulation. """
        return self.__time

    def __update_vehicles(self, soft_reset: bool = False):
        """
        Spawn new vehicles in the simulation based on when they appear in the dataset.
        Remove the vehicles that are dead.

        :param soft_reset: If True, get the new observation for the agent being evaluated.
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

            spawn_probability = self.SPAWN_PROBABILITY
            if random.random() < spawn_probability:
                agent_to_spawn = self._get_vehicle_to_spawn()
                add_agents[agent_to_spawn.UUID] = (agent_to_spawn, self.__policy_type)

        elif self.__spawn_method == "dataset_one":
            # We want to iterate through the dataset, and add one agent at a time that follows our policy. Then, spawn
            # those that are at the same time as the dataset as following the trajectory. Then, when the agent is dead,
            # remove it and add the next one, starting the time of the simulator back to the spawn time of the agent.

            if self.__agent_evaluated is None:
                a = self.__agents_to_add.keys() # TODO
                assert soft_reset , "Agent evaluated is None, but soft_reset (we want a new RL episode) is False"

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
            self._add_agent(self._create_policy_agent(agent, policy=policy))

        # Generate the first observation for the new agents
        obs_to_return = None
        for agent_id, _ in add_agents.items():
            agent = self.__agents[agent_id]
            obs = self._get_observation(agent, self.__state[agent_id])

            if soft_reset  and agent_id == self.__agent_evaluated:
                obs_to_return = obs
        return obs_to_return


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

    def __take_actions(self, ego_action: Tuple[float, float] = None):
        """
        Advance all agents by one time step.
        :param ego_action: only used it if the policy is "rl" for the ego vehicle (self.__agent_evaluated).
                            order is (acceleration, steering angle)
        """

        # A frame is a dictionary (agent_id, State)
        new_frame = {}

        # Given the above observations, choose an action and compute the next state
        for agent_id, agent in self.__agents.items():
            if agent is None and agent_id not in self.__dead_agents:
                continue

            assert len(agent.observation_trajectory) == len(agent.action_trajectory)+1 == len(agent.state_trajectory)

            if agent.policy == "follow_dataset":
                dataset_time_step = round((self.__time - agent.original_initial_time) / self.__dt)
                # Get the acceleration and steering angle from the dataset
                dataset_agent = self.__episode_agents[agent_id]
                deltas = ExtractObservationAction.extract_yaw_rate(agent=dataset_agent)
                acceleration = np.sqrt(float(dataset_agent.ax_vec[dataset_time_step]) ** 2 +
                                       float(dataset_agent.ay_vec[dataset_time_step]) ** 2)
                action = Action(acceleration=acceleration,
                                steer_angle=deltas[dataset_time_step])
            elif isinstance(agent.policy, BC):
                action = agent.next_action(history=agent.observation_trajectory)
            elif agent.policy == "rl":
                assert agent_id == self.__agent_evaluated, "Only the agent being evaluated can have policy 'rl'"
                action = Action(acceleration=ego_action[0], steer_angle=ego_action[1])
            else:
                raise NotImplementedError(f"Policy {agent.policy} not found.")

            # Use the bicycle model to find where the agent will be at t+1
            new_state = self._next_state(agent, current_state=self.__state[agent_id], action=action)

            goal_reached, truncated = False, False
            if agent.policy == "follow_dataset":

                if dataset_time_step + 1 < len(dataset_agent.x_vec):
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
                    goal_reached = True
            else:
                goal_reached = agent.reached_goal(new_state)
                truncated = agent.terminated(max_steps=1000)  # TODO: max_steps should be a parameter

            off_road = new_state.lane is None

            if off_road:
                self.__dead_agents[agent_id] = DeathCause.OFF_ROAD
            elif truncated:
                self.__dead_agents[agent_id] = DeathCause.TRUNCATED
            elif goal_reached:
                self.__dead_agents[agent_id] = DeathCause.GOAL_REACHED

            new_frame[agent_id] = new_state
            agent.add_action(action)
            agent.add_state(new_state)

        new_frame["time"] = self.__time + self.__dt
        self.__simulation_history.append(new_frame)
        self.__state = new_frame

        return self._get_current_observations(return_obs_for_aid=self.__agent_evaluated)

    def _get_current_observations(self, return_obs_for_aid: str = None):
        """
        :param return_obs_for_aid: id of the agent for which we want to return the observation
        :return:
        """
        # Compute the observation for each agent for the current state
        obs_to_return = None
        for agent_id, agent in self.__agents.items():

            if agent is None:
                continue

            obs = self._get_observation(agent=agent, state=self.__state[agent_id])
            if return_obs_for_aid is not None and agent_id == return_obs_for_aid:
                obs_to_return = obs

        return obs_to_return


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

        # TODO: bicycle model
        # acceleration = np.clip(action.acceleration, - agent.meta.max_acceleration, agent.meta.max_acceleration)
        #
        # speed = current_state.speed + acceleration * self.__dt
        # speed = max(0, speed)
        #
        # beta = np.arctan(agent._l_r * np.tan(action.steer_angle) / agent.meta.wheelbase)
        # d_position = np.array(
        #     [speed * np.cos(beta + current_state.heading),
        #      speed * np.sin(beta + current_state.heading)]
        # )
        #
        # center = np.array([current_state.position.x, current_state.position.y]) + d_position * self.__dt
        # d_theta = speed * np.tan(action.steer_angle) * np.cos(beta) / agent.meta.wheelbase
        # d_theta = np.clip(d_theta, - agent.meta.max_angular_vel, agent.meta.max_angular_vel)
        # heading = (current_state.heading + d_theta * self.__dt + np.pi) % (2 * np.pi) - np.pi

        # Unicycle model
        acceleration = np.clip(action.acceleration, - agent.meta.max_acceleration, agent.meta.max_acceleration)
        speed = current_state.speed + acceleration * self.__dt
        speed = max(0, speed)
        d_theta = action.steer_angle

        # update heading but respect the (-pi, pi) convention
        heading = (current_state.heading + d_theta * self.__dt + np.pi) % (2 * np.pi) - np.pi
        d_position = np.array([speed * np.cos(heading), speed * np.sin(heading)])
        center = np.array([current_state.position.x, current_state.position.y]) + d_position * self.__dt

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

        if state.lane is not None:
            # If it is, the agent went off the road.
            distance_left_lane_marking, distance_right_lane_marking = compute_distance_markings(state=state)

            # nearby_agents_features contains dx, dy, v, a, heading for each nearby agent
            # vehicles_nearby contains the agent object
            nearby_agents_features, vehicles_nearby = get_nearby_vehicles(agent=agent, state=state,
                                                                          all_agents=self.__agents)

            if collision_check(agent_state=self.__state[agent.agent_id], nearby_vehicles=vehicles_nearby) :
                self.__dead_agents[agent.agent_id] = DeathCause.COLLISION

        collision, off_road, truncated, reached_goal = False, False, False, False
        if agent.agent_id in self.__dead_agents:
            death_cause = self.__dead_agents[agent.agent_id]

            if death_cause == DeathCause.COLLISION:
                collision = True
            elif death_cause == DeathCause.OFF_ROAD:
                off_road = True
            elif death_cause == DeathCause.GOAL_REACHED:
                reached_goal = True
            elif death_cause == DeathCause.TRUNCATED:
                truncated = True
            else:
                raise ValueError(f"Death cause {death_cause} not found.")

        done = collision or off_road or truncated or reached_goal
        assert done or state.lane is not None, f"Agent {agent.agent_id} went off the road."

        if not done:
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

            obs = Observation(state=observation)

            # Compute the evaluation features for the agent
            if self.evaluation:
                agent.add_nearby_vehicles(vehicles_nearby)
                agent.add_distance_right_lane_marking(obs.get_feature("distance_right_lane_marking"))
                agent.add_distance_left_lane_marking(obs.get_feature("distance_left_lane_marking"))

            # Put the observation in a tuple, as the policy expects it
            obs = obs.get_tuple()
            agent.add_observation(obs)
        else:
            obs = None

        info = {"reached_goal": reached_goal, "collision": collision, "off_road": off_road, "truncated": truncated}

        return obs, info

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

                if self.__spawn_method == "dataset_one":
                    # Make the agent we are evaluating blue and all other agents grey
                    color = "blue" if agent_id == self.__agent_evaluated else "white"
                else:
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

        ani = FuncAnimation(fig, update, frames=self.__simulation_history, repeat=True,
                            interval=self.__fps)  # TODO: interval=self.__fps
        plt.show()

    @property
    def dt(self):
        return self.__dt

    @property
    def evaluator(self):
        return self.__eval

    @property
    def simulation_history(self):
        return self.__simulation_history  # todo: remove this


if __name__ == "__main__":

    # TODO: loop over episodes
    # data_loader = DatasetDataLoader(f"scenarios/configs/appershofen.json")
    # data_loader.load()
    # episodes = data_loader.scenario.episodes
    # episode = episodes[0]

    episode_name = "hw-a9-appershofen-001-d8087340-8287-46b6-9612-869b09e68448"
    path_to_dataset_folder = get_path_to_automatum_scenario(episode_name)
    dataset = droneDataset(path_to_dataset_folder)
    dyn_world = dataset.dynWorld
    dt = dyn_world.delta_t

    scenario_name = episode_name.split("-")[2]
    config = ScenarioConfig.load(get_config_path(scenario_name))
    data_loader = DatasetScenario(config)
    episode = data_loader.load_episode(episode_id=episode_name)

    scenario_map = Map.parse_from_opendrive(get_path_to_automatum_map(episode_name))

    sim = Sim4ADSimulation(scenario_map, episode_agents=episode.agents, dt=dt, spawn_method="dataset_one",
                           policy_type="bc-all-obs")
    sim.full_reset()

    simulation_length = 100  # seconds

    # TODO: use tqdm
    for _ in range(int(np.floor(simulation_length / dt))):
        sim.step()

    # Remove all agents left in the simulation.
    sim.kill_all_agents()

    sim.replay_simulation()

    print("Simulation done!")

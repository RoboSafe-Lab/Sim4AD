"""
This file is used to run a given policy on a given scenario and generate the trajectories of the vehicles in the
scenario.

The basic structure was initially vaguely based on https://github.com/uoe-agents/IGP2/blob/main/igp2/simplesim/simulation.py
"""
import json
import logging
import random
from collections import defaultdict
from copy import deepcopy
from typing import Tuple, List, Dict, Any, Union

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from openautomatumdronedata.dataset import droneDataset
from shapely import Point, LineString
from stable_baselines3 import SAC
from tqdm import tqdm

from baselines.bc_baseline import BCBaseline as BC
from baselines.idm import IDM
from sim4ad.offlinerlenv.td3bc_automatum import TD3_BC_TrainerLoader, TrainConfig
from extract_observation_action import ExtractObservationAction
from sim4ad.common_constants import MISSING_NEARBY_AGENT_VALUE
from sim4ad.data import DatasetDataLoader, ScenarioConfig, DatasetScenario
from sim4ad.irlenv.vehicle.behavior import IDMVehicle
from sim4ad.opendrive import plot_map, Map
from sim4ad.path_utils import get_path_to_automatum_scenario, get_path_to_automatum_map, \
    get_config_path, get_path_offlinerl_model
from sim4ad.util import Box
from simulator.policy_agent import PolicyAgent, DummyRandomAgent
from simulator.state_action import State, Action, Observation
from simulator.simulator_util import DeathCause, get_nearby_vehicles, compute_distance_markings, collision_check
from simulator.simulator_util import PositionNearbyAgent as PNA
from evaluation.evaluation_functions import EvaluationFeaturesExtractor

from sim4ad.common_constants import DEFAULT_DECELERATION_VALUE

logger = logging.getLogger(__name__)


class Sim4ADSimulation:

    def __init__(self,
                 episode_name: Union[List[str], str],
                 policy_type: str = "bc",
                 simulation_name: str = "sim4ad_simulation",
                 spawn_method: str = "dataset",
                 evaluation: bool = True,
                 clustering: str = "all"):

        """ Initialise new simulation.

        Args:
            policy_type: The type of policy to use.
            simulation_name: The name of the simulation.
            spawn_method: The method to spawn the vehicles in the simulation. Either "dataset_all", "dataset_one" or "random".
        """

        self.episode_idx = 0  # Index of the episode we are currently evaluating
        self.__all_episode_names = episode_name
        self.will_be_done_next = False
        self.done_full_cycle = False  # If iterate once through all agents in all episodes
        self.clustering = clustering
        self.__load_datasets()
        self.__fps = np.round(1 / self.__dt)

        self.evaluation = evaluation
        self.SPAWN_PROBABILITY = 0.02

        self.__simulation_name = simulation_name
        self.__eval = EvaluationFeaturesExtractor(sim_name=simulation_name)

        # dataset_all: agents are spawned at the time they appear in the dataset, but are controlled by the policy.
        # random: agents are spawned at random times and positions.
        # dataset-one: all but one agent follow the dataset, the other one, is controlled by the policy.
        assert spawn_method in ["dataset_all", "random", "dataset_one"], f"Spawn method {spawn_method} not found."

        self.__spawn_method = spawn_method

        assert policy_type in ["follow_dataset", "rl", "idm", "offlinerl"] or "bc" in policy_type.lower() \
               or "sac" in policy_type.lower(), f"Policy type {policy_type} not found."
        if policy_type == "follow_dataset":
            assert spawn_method != "random", "Policy type 'follow_dataset' is not compatible with 'random' spawn"

        self.__policy_type = policy_type

        # Also in load_dataset
        self.__time = 0
        self.__state = {}
        self.__agents = {}
        self.__agents_to_add = deepcopy(self.__episode_agents)  # Agents that have not been added to the simulation yet.
        if self.clustering != "all":
            self.__agents_to_add = self.cluster_agents(self.__episode_agents)

        self.__simulation_history = []  # History of frames (agent_id, State) of the simulation.
        self.__last_agent_id = 0  # ID when creating a new random agent
        # Dictionary (agent_id, DeathCause) of agents that have been removed from the simulation.
        self.__dead_agents = {}
        self.__agent_evaluated = None  # If we spawn_method is "dataset_one", then this is the agent we are evaluating.

    def cluster_agents(self, agents: Dict[str, droneDataset]):
        """Return agents of a specific cluster"""
        episode_name = self.__all_episode_names[self.episode_idx - 1]
        scenario_name = episode_name.split("-")[2]
        with open(f"scenarios/configs/{scenario_name}_drivingStyle.json", "rb") as f:
            import json
            clusterings = json.load(f)
            return {k: v for k, v in agents.items() if self.clustering == clusterings[f"{episode_name}/{k}"]}

    def seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)

    def _add_agent(self, agent, policy: str):
        """ Add a new agent to the simulation.

        Args:
            agent: Agent to add.
        """

        if policy == "idm":
            new_agent = self._create_policy_agent(agent, policy="follow_dataset")
            new_agent.idm.activate(v0=new_agent.state.speed)
        else:
            new_agent = self._create_policy_agent(agent, policy=policy)

        if new_agent.agent_id in self.__agents \
                and self.__agents[new_agent.agent_id] is not None:
            raise ValueError(f"Agent with ID {new_agent.agent_id} already exists.")

        if self.__spawn_method in ["dataset_all", "random"] or new_agent.agent_id == self.__agent_evaluated:
            self.__state[new_agent.agent_id] = new_agent.initial_state
        elif self.__spawn_method == "dataset_one":
            # Spawn the vehicle at the state it was in the dataset at the time of the simulation.
            new_agent_original = self.__episode_agents[new_agent.agent_id]
            current_state = self.__get_original_current_state(new_agent_original)

            # Check if safe to spawn or if maybe there is another vehicle in that position, such as a vehicle that is
            # controlled by IDM and therefore deviated from what the corresponding vehicle did in the dataset or the
            # ego vehicle that is controlled by the vehicle deviated from the dataset.
            safe_to_spawn = self.__safe_to_spawn(position=current_state.position, width=new_agent.meta.width,
                                                 length=new_agent.meta.length, heading=current_state.heading,
                                                 current_lane=current_state.lane)

            if safe_to_spawn:
                # Add it to the current state if
                self.__state[new_agent.agent_id] = current_state
            else:
                self.__dead_agents[new_agent.agent_id] = DeathCause.COLLISION
        else:
            raise ValueError(f"Spawn method {self.__spawn_method} not found.")

        self.__agents[new_agent.agent_id] = new_agent

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

        if "bc" in policy.lower():
            return BC(name=policy, evaluation=True)
        elif "sac" in policy.lower():
            return SAC.load(policy)
        elif policy == "follow_dataset":
            return "follow_dataset"
        elif policy == "rl":
            return "rl"
        elif policy == 'offlinerl':
            model_path = get_path_offlinerl_model()
            trainer_loader = TD3_BC_TrainerLoader(TrainConfig)
            return trainer_loader.load_model(model_path)
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
            raise ValueError("Agent should be a dataset agent, not a PolicyAgent.")
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
            death_cause: reason for death
        """

        agent_removed = self.__agents.pop(agent_id)

        # Save the agent's trajectory for evaluation
        if self.__spawn_method in ["dataset_all", "random"] or agent_id == self.__agent_evaluated:
            self.__eval.save_trajectory(agent=agent_removed, death_cause=death_cause,
                                        episode_id=self._get_current_episode_id())

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
        self.will_be_done_next = False
        self.done_full_cycle = False

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

        initial_obs = None
        info = None
        if len(self.__agents_to_add.keys()) > 0:
            initial_obs, info = self.__update_vehicles(soft_reset=True)

        if initial_obs is None:
            self.__change_episode()
            initial_obs, info = self.__update_vehicles(soft_reset=True)

        return initial_obs, info

    def step(self, action: Tuple[float, float] = None, return_done: bool = False):
        """
        Advance simulation by one time step.
        If action is provided, we will use that action for the agent being evaluated (only if policy is "rl").
        Order is (acceleration, steering angle).
        """

        logger.debug(f"Simulation time {self.__time}")
        self.__update_vehicles()
        ego_obs = self.__take_actions(ego_action=action)
        self.__time += self.__dt

        if return_done:
            return self.done_full_cycle
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

        self.__remove_dead_agents()

        # SELECT AGENTS TO ADD
        add_agents = {}

        # if finished the current episode, move to the next one
        if self.__spawn_method in ["dataset_all", "dataset_one"] and self.__policy_type != "rl":
            if len(self.__agents_to_add) == 0:
                self.__change_episode()

        if self.__spawn_method == "dataset_all":
            for agent_id, agent in self.__agents_to_add.items():
                if (self.__time - agent.time[0]) > -1e-6 and (agent.time[-1] - self.time) > -1e-6:
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
                if self.__policy_type == "rl":
                    assert soft_reset, "Agent evaluated is None, but soft_reset (we want a new RL episode) is False"

                # Get the first agent in the dataset
                self.__agent_evaluated = list(self.__agents_to_add.keys())[0]
                self.__time = self.__agents_to_add[self.__agent_evaluated].time[0]

            # Spawn all vehicles alive at the current time.
            for agent_id, agent in self.__episode_agents.items():
                if (self.__time - agent.time[0]) > -1e-6 and (agent.time[-1] - self.time) > -1e-6:
                    if agent_id not in self.__agents:
                        if agent_id == self.__agent_evaluated:
                            add_agents[agent_id] = (agent, self.__policy_type)
                        else:
                            add_agents[agent_id] = (agent, "follow_dataset")
        else:
            raise ValueError(f"Spawn method {self.__spawn_method} not found.")

        # ADD AGENTS
        for agent_id, (agent, policy) in add_agents.items():
            self._add_agent(agent=agent, policy=policy)

        # Generate the first observation for the new agents
        obs = None
        info = {}
        for agent_id, _ in add_agents.items():

            if agent_id in self.__dead_agents:
                # An agent could be dead if, for example, we tried to spawn it but there was another vehicle
                # in the same position. This should not happen as when we spawn the agent evaluated, the
                # other vehicles should all reset to the initial state in the dataset where there were no collisions.
                assert agent_id != self.__agent_evaluated, "Agent evaluated is dead, but it should not be."
                continue

            agent = self.__agents[agent_id]
            obs_, info_ = self._get_observation(agent, self.__state[agent_id], info)

            if soft_reset and agent_id == self.__agent_evaluated:
                obs, info = obs_, info_

        # Remove any agents that couldn't be added due to collisions at spawn.
        self.__remove_dead_agents()

        return obs, info

    def __remove_dead_agents(self):
        # REMOVE DEAD AGENTS
        for agent_id, death_cause in self.__dead_agents.items():
            self.remove_agent(agent_id, death_cause)
            logger.debug(f"Agent {agent_id} has been removed from the simulation for {death_cause} at t={self.time}.")

        # If we removed the agent we are evaluating, and we are in "dataset_one" mode, then we need to remove all.
        if self.__spawn_method == "dataset_one" and self.__agent_evaluated in self.__dead_agents:
            self.__agents = {}
            self.__agent_evaluated = None

        self.__dead_agents = {}

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

            safe_to_spawn = self.__safe_to_spawn(position=spawn_position, width=spawn_width, length=spawn_length,
                                                 heading=spawn_heading, current_lane=spawn_lane)

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

    def __safe_to_spawn(self, position: np.array, width: float, length: float, heading: float, current_lane) -> bool:
        # Check if the position is free
        bbox = Box(center=position, length=length, width=width, heading=heading)

        safe_to_spawn = True
        for agent in self.__agents.values():
            if agent.state.bbox.overlaps(bbox):
                safe_to_spawn = False
                break

        # Check if there is a vehicle in the same position as the vehicle we are spawning and check that the vehicle
        # is on a drivable road (i.e., its lane is not None).
        return safe_to_spawn and current_lane is not None

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

            assert len(agent.observation_trajectory) == len(agent.action_trajectory) + 1 == len(agent.state_trajectory)

            if agent.policy == "follow_dataset":
                action, new_state = self.__handle_follow_dataset(agent, agent_id)
            elif isinstance(agent.policy, BC) or isinstance(agent.policy, SAC):
                action = agent.next_action(history=agent.observation_trajectory)
                # Use the bicycle model to find where the agent will be at t+1
                new_state = self._next_state(agent, current_state=self.__state[agent_id], action=action)
            elif agent.policy == "rl":
                assert agent_id == self.__agent_evaluated, "Only the agent being evaluated can have policy 'rl'"
                action = Action(acceleration=ego_action[0], steer_angle=ego_action[1])
                new_state = self._next_state(agent, current_state=self.__state[agent_id], action=action)
            else:
                raise NotImplementedError(f"Policy {agent.policy} not found.")

            goal_reached, truncated = self.__check_goal_and_termination(agent, new_state, agent_id)

            if new_state.lane is None:
                self.__dead_agents[agent_id] = DeathCause.OFF_ROAD
            elif truncated:
                self.__dead_agents[agent_id] = DeathCause.TRUNCATED
            elif goal_reached:
                self.__dead_agents[agent_id] = DeathCause.GOAL_REACHED

            new_frame[agent_id] = new_state
            agent.add_action(action)
            agent.add_state(new_state)

        new_frame["time"] = self.__time + self.__dt
        new_frame["evaluated_agent"] = self.__agent_evaluated
        self.__simulation_history.append(new_frame)
        self.__state = new_frame

        return self._get_current_observations(return_obs_for_aid=self.__agent_evaluated)

    def __check_goal_and_termination(self, agent, new_state, agent_id):
        if agent.policy == "follow_dataset" and not agent.idm.activated():
            dataset_agent = self.__episode_agents[agent_id]
            dataset_time_step = round((self.__time - agent.original_initial_time) / self.__dt)
            if dataset_time_step + 1 >= len(dataset_agent.x_vec):
                return True, False
        goal_reached = agent.reached_goal(new_state)
        truncated = agent.terminated(max_steps=1000)
        return goal_reached, truncated

    def __handle_follow_dataset(self, agent, agent_id):
        dataset_agent = self.__episode_agents[agent_id]
        # Check if the current gap with the vehicle in front is less than the minimum gap
        vehicle_in_front = agent.last_vehicle_in_front_ego()
        dataset_time_step = round((self.__time - agent.original_initial_time) / self.__dt)

        if vehicle_in_front is not None and not agent.idm.activated():
            self.__check_and_activate_idm(agent, dataset_agent, vehicle_in_front, dataset_time_step)

        if agent.idm.activated():
            action = agent.idm.compute_idm_action(
                state_i=agent.state, agent_in_front=vehicle_in_front,
                agent_meta=agent.meta,
                previous_state_i=agent.state_trajectory[-2] if len(agent.state_trajectory) > 1 else agent.state
            )
            new_state = self._next_state(agent, current_state=self.__state[agent_id], action=action)
        else:
            deltas = ExtractObservationAction.extract_yaw_rate(agent=dataset_agent)
            acceleration = np.sqrt(float(dataset_agent.ax_vec[dataset_time_step]) ** 2 +
                                   float(dataset_agent.ay_vec[dataset_time_step]) ** 2)
            action = Action(acceleration=acceleration, steer_angle=deltas[dataset_time_step])

            if dataset_time_step + 1 < len(dataset_agent.x_vec):
                position = Point(dataset_agent.x_vec[dataset_time_step + 1], dataset_agent.y_vec[dataset_time_step + 1])
                speed = np.sqrt(float(dataset_agent.vx_vec[dataset_time_step + 1]) ** 2 +
                                float(dataset_agent.vy_vec[dataset_time_step + 1]) ** 2)
                heading = dataset_agent.psi_vec[dataset_time_step + 1]
                acceleration = np.sqrt(float(dataset_agent.ax_vec[dataset_time_step + 1]) ** 2 +
                                       float(dataset_agent.ay_vec[dataset_time_step + 1]) ** 2)
                new_state = State(time=self.__time + self.__dt, position=position, speed=speed,
                                  acceleration=acceleration,
                                  heading=heading, lane=agent.state.lane, agent_width=agent.meta.width,
                                  agent_length=agent.meta.length)
            else:
                new_state = agent.state
                self.__dead_agents[agent_id] = DeathCause.GOAL_REACHED

        return action, new_state

    def __check_and_activate_idm(self, agent, dataset_agent, vehicle_in_front, dataset_time_step):
        # agent died at the previous time step
        vehicle_front_not_alive_anymore = vehicle_in_front["agent_id"] not in self.__agents
        if vehicle_front_not_alive_anymore:
            return

        bbox_i = LineString(agent.state.bbox.boundary)
        vehicle_in_front_simulator = self.__agents[vehicle_in_front["agent_id"]]
        bbox_j = LineString(vehicle_in_front_simulator.state.bbox.boundary)
        gap = bbox_i.distance(bbox_j)

        if gap < agent.idm.s0 and dataset_time_step + 1 < len(dataset_agent.x_vec):
            self.__compare_and_activate_idm_if_needed(agent, vehicle_in_front, bbox_i, gap,
                                                      vehicle_in_front_simulator)

    def __compare_and_activate_idm_if_needed(self, agent, vehicle_in_front, bbox_i, gap, vehicle_in_front_simulator):
        expected_v_front = self.__episode_agents[vehicle_in_front["agent_id"]]
        time_j = round((self.__time - expected_v_front.time[0]) / self.__dt)

        try:
            expected_j_bbox = Box(
                center=Point(expected_v_front.x_vec[time_j], expected_v_front.y_vec[time_j]),
                length=expected_v_front.length, width=expected_v_front.width,
                heading=expected_v_front.psi_vec[time_j]
            )
            expected_gap = bbox_i.distance(LineString(expected_j_bbox.boundary))

            if (gap - expected_gap) < 1e-4:
                debug = False
                if debug:
                    fig, ax = plt.subplots()
                    plot_map(self.__scenario_map, markings=True,
                             hide_road_bounds_in_junction=True, ax=ax)
                    for ajd, aj in self.__agents.items():
                        color = "red" if ajd == self.__agent_evaluated else "blue"
                        plt.plot(aj.state.position.x, aj.state.position.y, "o")
                        # blot bonding
                        # Plot the bounding box of the agent
                        bbox = aj.state.bbox.boundary
                        # repeat the first point to create a 'closed loop'
                        bbox = [*bbox, bbox[0]]
                        ax.plot([point[0] for point in bbox], [point[1] for point in bbox],
                                color=color)
                    # plot the expected bounding box of the front vehicle in white
                    bbox = expected_j_bbox.boundary
                    # repeat the first point to create a 'closed loop'
                    bbox = [*bbox, bbox[0]]
                    ax.plot([point[0] for point in bbox], [point[1] for point in bbox],
                            color="green")
                    plt.show()
                self.__activate_idm(agent, vehicle_in_front, agent.state.speed)

        except IndexError:
            # the vehicle in front is not following the dataset, and thus we cannot compare
            # its position to the original one.
            # The gap is critical and different from what it should be
            if vehicle_in_front_simulator.policy not in ["follow_dataset"]:
                self.__activate_idm(agent, vehicle_in_front, agent.state.speed)

    def __activate_idm(self, agent, vehicle_in_front, speed):
        agent.idm.activate(v0=speed)
        self.__agents[self.__agent_evaluated].add_interference(agent_id=vehicle_in_front["agent_id"])

    def _get_current_observations(self, return_obs_for_aid: str = None):
        """
        :param return_obs_for_aid: id of the agent for which we want to return the observation
        :return:
        """
        # Compute the observation for each agent for the current state
        obs = None
        info = {}
        for agent_id, agent in self.__agents.items():

            if agent is None:
                continue

            obs_, info_ = self._get_observation(agent=agent, state=self.__state[agent_id], info=info)
            if return_obs_for_aid is not None and agent_id == return_obs_for_aid:
                obs, info = obs_, info_

        return obs, info

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

        # bicycle model
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
        if agent.idm.activated():
            # we move along the midline of the road, rather than following the current one
            new_lane = current_state.lane
            center = np.array([current_state.position.x, current_state.position.y]) + np.array(
                [speed * np.cos(current_state.heading), speed * np.sin(current_state.heading)]) * self.__dt
            center_ds = new_lane.distance_at(Point(center[0], center[1]))
            center = new_lane.point_at(center_ds)
            heading = new_lane.get_heading_at(center_ds)

        else:
            d_theta = action.steer_angle

            # update heading but respect the (-pi, pi) convention
            heading = (current_state.heading + d_theta * self.__dt + np.pi) % (2 * np.pi) - np.pi
            d_position = np.array([speed * np.cos(heading), speed * np.sin(heading)])
            center = np.array([current_state.position.x, current_state.position.y]) + d_position * self.__dt
            new_lane = self.__scenario_map.best_lane_at(center, heading)

        return State(time=self.time + self.dt, position=center, speed=speed, acceleration=acceleration,
                     heading=heading, lane=new_lane, agent_width=agent.meta.width, agent_length=agent.meta.length)

    def _check_death_cause(self, agent):
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

        return collision, off_road, truncated, reached_goal

    def _handle_none_lane(self, agent, off_road):
        if not off_road:
            plot_map(self.__scenario_map, markings=True, hide_road_bounds_in_junction=True)
            for agent_id, agent in self.__agents.items():
                color = "red" if agent_id == self.__agent_evaluated else "blue"
                plt.plot(agent.state.position.x, agent.state.position.y, "o", color=color)
                bbox = agent.state.bbox.boundary + [agent.state.bbox.boundary[0]]
                plt.plot([point[0] for point in bbox], [point[1] for point in bbox], color=color)
            plt.show()
            print()

        assert off_road, f"Agent {agent.agent_id} went off the road but off_road is False. Death cause: " \
                         f"{self.__dead_agents.get(agent.agent_id)}"

    def _get_observation(self, agent: PolicyAgent, state: State, info: dict) -> Tuple[Observation, dict]:
        """
        Get the current observation of the agent.

        :param agent: The agent.
        :return: The observation and the nearby agents.
        """
        # Initialize variables
        nearby_agents_features, vehicles_nearby = {}, []
        distance_left_lane_marking, distance_right_lane_marking = None, None

        if state.lane is not None:
            # If it is, the agent went off the road.
            distance_left_lane_marking, distance_right_lane_marking = compute_distance_markings(state=state)

            # nearby_agents_features contains dx, dy, v, a, heading for each nearby agent
            # vehicles_nearby contains the agent object
            nearby_agents_features, vehicles_nearby = get_nearby_vehicles(agent=agent, state=state,
                                                                          all_agents=self.__agents)

            if collision_check(agent_state=self.__state[agent.agent_id], nearby_vehicles=vehicles_nearby):
                self.__dead_agents[agent.agent_id] = DeathCause.COLLISION

        collision, off_road, truncated, reached_goal = self._check_death_cause(agent)

        # for debug
        if state.lane is None:
            self._handle_none_lane(agent, off_road)

        done = collision or off_road or truncated or reached_goal
        info.update({"reached_goal": reached_goal, "collision": collision, "off_road": off_road, "truncated": truncated})
        if not done:
            observation = self._build_observation(state, nearby_agents_features, distance_left_lane_marking,
                                                  distance_right_lane_marking)

            obs = Observation(state=observation)
            # Compute the evaluation features for the agent
            assert self.evaluation or self.__policy_type != "rl", "We need these features to use the IRL reward"
            if self.evaluation:
                nearby_vehicles = agent.add_nearby_vehicles(vehicles_nearby)
                agent.add_distance_right_lane_marking(obs.get_feature("distance_right_lane_marking"))
                agent.add_distance_left_lane_marking(obs.get_feature("distance_left_lane_marking"))
                d_midline = state.lane.midline.distance(state.position)
                agent.add_distance_midline(d_midline)

                # Compute the features needed to use the IRL reward (and evaluation)
                self._update_info(agent, nearby_vehicles, state, observation, done, info)

            # Put the observation in a tuple, as the policy expects it
            obs = Observation(state=observation).get_tuple()
            agent.add_observation(obs)
        else:
            obs = None

        return obs, info

    def _update_info(self, agent, nearby_vehicles, state, observation, done, info):
        ax, ay = agent.compute_current_lat_lon_acceleration()
        long_jerk = agent.compute_current_long_jerk()
        _, tths = self.evaluator.compute_ttc_tth(agent, state=state, nearby_vehicles=nearby_vehicles,
                                                 episode_id=None, add=False)
        thw_front, thw_rear = tths[PNA.CENTER_IN_FRONT], tths[PNA.CENTER_BEHIND]

        info.update({
            "ego_speed": state.speed,
            "ego_long_acc": ax,
            "ego_lat_acc": ay,
            "ego_long_jerk": long_jerk,
            "thw_front": thw_front,
            "thw_rear": thw_rear,
            "induced_deceleration": DEFAULT_DECELERATION_VALUE
        })

        if self.__policy_type == "rl":
            induced_deceleration = 0
            behind_ego = {
                "": MISSING_NEARBY_AGENT_VALUE
            }  # There is no vehicle behind the ego, as per the check later in the code
            behind_ego_missing = sum([x == MISSING_NEARBY_AGENT_VALUE for x in behind_ego.values()]) == len(behind_ego)

            if not done and not behind_ego_missing:
                ego_rear_d = np.sqrt((observation["behind_ego_rel_dx"]) ** 2 + (observation["behind_ego_rel_dy"]) ** 2)
                rear_position = np.array([state.position.x + observation["behind_ego_rel_dx"],
                                          state.position.y + observation["behind_ego_rel_dy"]])
                induced_deceleration = self.compute_induced_deceleration(
                    ego_v=state.speed,
                    ego_length=agent.meta.length,
                    rear_agent=behind_ego,
                    ego_rear_d=ego_rear_d,
                    rear_a=observation["behind_ego_rel_a"],
                    rear_position=rear_position
                )
                info["induced_deceleration"] = induced_deceleration

        return info

    def compute_induced_deceleration(self, ego_v, ego_length, rear_agent, ego_rear_d, rear_a,
                                     rear_position) -> float:
        """

        :param ego_v:               velocity of the ego vehicle
        :param ego_length:          length of the ego vehicle
        :param ego_rear_d:          distance between the ego vehicle and the rear vehicle
        :param rear_agent:
        :param rear_a:              acceleration of the rear vehicle
        :param rear_position:       position of the rear vehicle
        :return:
        """

        class DummyVehicle:
            def __init__(self, v, length):
                self.velocity = [v]
                self.LENGTH = length

        # Compute the safe distance according to IDM
        idm = IDMVehicle(scenario_map=self.__scenario_map, position=rear_position, heading=rear_agent["heading"],
                         velocity=rear_agent["speed"])
        safe_d = idm.desired_gap(ego_vehicle=DummyVehicle(rear_agent["speed"], rear_agent["length"]),
                                 front_vehicle=DummyVehicle(ego_v, ego_length))

        # If the distance is less than the minimum distance, then return the deceleration of the vehicle behind
        if ego_rear_d < safe_d:
            if rear_a < 0:
                return abs(rear_a)

        return 0.

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
            evaluated_agent = frame['evaluated_agent']
            for idx, (agent_id, state) in enumerate(frame.items()):
                if agent_id in ["time", "evaluated_agent"]:
                    continue

                if self.__spawn_method == "dataset_one":
                    # Make the agent we are evaluating blue and all other agents grey
                    color = "blue" if agent_id == evaluated_agent else "white"
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

                position = np.array([state.position.x, state.position.y])
                ax.plot(*position, marker=".", color=color)

                # Plot the bounding box of the agent
                bbox = state.bbox.boundary
                # repeat the first point to create a 'closed loop'
                bbox = [*bbox, bbox[0]]
                ax.plot([point[0] for point in bbox], [point[1] for point in bbox], color=color)

            ax.set_title(f"Simulation Time: {time}")

        ani = FuncAnimation(fig, update, frames=self.__simulation_history, repeat=True,
                            interval=self.__fps)
        plt.show()

    def __load_datasets(self):

        if self.will_be_done_next:
            self.done_full_cycle = True

        if isinstance(self.__all_episode_names, list):
            episode_name = self.__all_episode_names[self.episode_idx]
            self.episode_idx += 1
            if self.episode_idx == len(self.__all_episode_names):
                self.episode_idx = 0
                self.will_be_done_next = True
        else:
            episode_name = self.__all_episode_names
            self.will_be_done_next = True

        path_to_dataset_folder = get_path_to_automatum_scenario(episode_name)
        dataset = droneDataset(path_to_dataset_folder)
        dyn_world = dataset.dynWorld
        self.__dt = dyn_world.delta_t

        self.__scenario_name = episode_name.split("-")[2]
        config = ScenarioConfig.load(get_config_path(self.__scenario_name))
        data_loader = DatasetScenario(config)
        episode = data_loader.load_episode(episode_id=episode_name)

        self.__scenario_map = Map.parse_from_opendrive(get_path_to_automatum_map(episode_name))
        self.__episode_agents = episode.agents

        self.__time = 0
        self.__state = {}
        self.__agents = {}
        self.__agents_to_add = deepcopy(self.__episode_agents)  # Agents that have not been added to the simulation yet.

        if self.clustering != "all":
            self.__episode_agents = self.cluster_agents(self.__episode_agents)
        self.__simulation_history = []  # History of frames (agent_id, State) of the simulation.
        self.__last_agent_id = 0  # ID when creating a new random agent
        # Dictionary (agent_id, DeathCause) of agents that have been removed from the simulation.
        self.__dead_agents = {}
        self.__agent_evaluated = None  # If we spawn_method is "dataset_one", then this is the agent we are evaluating.

    @staticmethod
    def _build_observation(state, nearby_agents_features, distance_left_lane_marking,
                           distance_right_lane_marking):
        return {
            "speed": state.speed,
            "heading": state.heading,
            "distance_left_lane_marking": distance_left_lane_marking,
            "distance_right_lane_marking": distance_right_lane_marking,
            "front_ego_rel_dx": nearby_agents_features.get(PNA.CENTER_IN_FRONT, {}).get("rel_dx", 0),
            "front_ego_rel_dy": nearby_agents_features.get(PNA.CENTER_IN_FRONT, {}).get("rel_dy", 0),
            "front_ego_rel_speed": nearby_agents_features.get(PNA.CENTER_IN_FRONT, {}).get("speed", 0) - state.speed,
            "front_ego_rel_a": nearby_agents_features.get(PNA.CENTER_IN_FRONT, {}).get("a", 0) - state.acceleration,
            "front_ego_heading": nearby_agents_features.get(PNA.CENTER_IN_FRONT, {}).get("heading", 0),
            "behind_ego_rel_dx": nearby_agents_features.get(PNA.CENTER_BEHIND, {}).get("rel_dx", 0),
            "behind_ego_rel_dy": nearby_agents_features.get(PNA.CENTER_BEHIND, {}).get("rel_dy", 0),
            "behind_ego_rel_speed": nearby_agents_features.get(PNA.CENTER_BEHIND, {}).get("speed", 0) - state.speed,
            "behind_ego_rel_a": nearby_agents_features.get(PNA.CENTER_BEHIND, {}).get("a", 0) - state.acceleration,
            "behind_ego_heading": nearby_agents_features.get(PNA.CENTER_BEHIND, {}).get("heading", 0),
            "front_left_rel_dx": nearby_agents_features.get(PNA.LEFT_IN_FRONT, {}).get("rel_dx", 0),
            "front_left_rel_dy": nearby_agents_features.get(PNA.LEFT_IN_FRONT, {}).get("rel_dy", 0),
            "front_left_rel_speed": nearby_agents_features.get(PNA.LEFT_IN_FRONT, {}).get("speed", 0) - state.speed,
            "front_left_rel_a": nearby_agents_features.get(PNA.LEFT_IN_FRONT, {}).get("a", 0) - state.acceleration,
            "front_left_heading": nearby_agents_features.get(PNA.LEFT_IN_FRONT, {}).get("heading", 0),
            "behind_left_rel_dx": nearby_agents_features.get(PNA.LEFT_BEHIND, {}).get("rel_dx", 0),
            "behind_left_rel_dy": nearby_agents_features.get(PNA.LEFT_BEHIND, {}).get("rel_dy", 0),
            "behind_left_rel_speed": nearby_agents_features.get(PNA.LEFT_BEHIND, {}).get("speed", 0) - state.speed,
            "behind_left_rel_a": nearby_agents_features.get(PNA.LEFT_BEHIND, {}).get("a", 0) - state.acceleration,
            "behind_left_heading": nearby_agents_features.get(PNA.LEFT_BEHIND, {}).get("heading", 0),
            "front_right_rel_dx": nearby_agents_features.get(PNA.RIGHT_IN_FRONT, {}).get("rel_dx", 0),
            "front_right_rel_dy": nearby_agents_features.get(PNA.RIGHT_IN_FRONT, {}).get("rel_dy", 0),
            "front_right_rel_speed": nearby_agents_features.get(PNA.RIGHT_IN_FRONT, {}).get("speed", 0) - state.speed,
            "front_right_rel_a": nearby_agents_features.get(PNA.RIGHT_IN_FRONT, {}).get("a", 0) - state.acceleration,
            "front_right_heading": nearby_agents_features.get(PNA.RIGHT_IN_FRONT, {}).get("heading", 0),
            "behind_right_rel_dx": nearby_agents_features.get(PNA.RIGHT_BEHIND, {}).get("rel_dx", 0),
            "behind_right_rel_dy": nearby_agents_features.get(PNA.RIGHT_BEHIND, {}).get("rel_dy", 0),
            "behind_right_rel_speed": nearby_agents_features.get(PNA.RIGHT_BEHIND, {}).get("speed", 0) - state.speed,
            "behind_right_rel_a": nearby_agents_features.get(PNA.RIGHT_BEHIND, {}).get("a", 0) - state.acceleration,
            "behind_right_heading": nearby_agents_features.get(PNA.RIGHT_BEHIND, {}).get("heading", 0)
        }

    def __change_episode(self):
        self.__load_datasets()

    @property
    def dt(self):
        return self.__dt

    @property
    def evaluator(self):
        return self.__eval

    @property
    def agents(self):
        return self.__agents

    @property
    def simulation_history(self):
        return self.__simulation_history

    def _get_current_episode_id(self):
        # We need -1 because we increment the episode_idx before using it.
        return self.__all_episode_names[self.episode_idx - 1] if isinstance(self.__all_episode_names,
                                                                            list) else self.__all_episode_names

    @property
    def agents_to_add(self):
        return self.__agents_to_add


if __name__ == "__main__":

    ep_name = ["hw-a9-appershofen-001-d8087340-8287-46b6-9612-869b09e68448",
               "hw-a9-appershofen-002-2234a9ae-2de1-4ad4-9f43-65c2be9696d6"]

    spawn_method = "dataset_one"
    # "bc-all-obs-5_pi_cluster_Aggressive"  # "bc-all-obs-1.5_pi" "idm"  "offlinerl"
    policy_type = "sac"
    clustering = "all"
    sim = Sim4ADSimulation(episode_name=ep_name, spawn_method=spawn_method, policy_type=policy_type,
                           clustering=clustering)
    sim.full_reset()

    # done = False # TODO: uncomment this to run until we use all vehicles
    # while not done:
    #     assert spawn_method != "random", "we will never finish!"
    #     done = sim.step(return_done=True)

    simulation_length = 50  # seconds
    for _ in tqdm(range(int(np.floor(simulation_length / sim.dt)))):
        sim.step()

    # Remove all agents left in the simulation.
    sim.kill_all_agents()

    sim.replay_simulation()

    print("Simulation done!")

"""
This file is used to run a given policy on a given scenario and generate the trajectories of the vehicles in the scenario.
"""
import logging
import random
from collections import defaultdict
from copy import deepcopy
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from shapely import Point, LineString
from shapely.ops import split
from tqdm import tqdm

from baselines.bc_baseline import PolicyNetwork as BC
from sim4ad.common_constants import MISSING_NEARBY_AGENT_VALUE
from sim4ad.data import DatasetDataLoader, Episode
from sim4ad.opendrive import plot_map, Lane, Map
from simulator.policy_agent import PolicyAgent
from simulator.state_action import State, Action, Observation
from simulator_util import DeathCause

logger = logging.getLogger(__name__)


# TODO: the basic structure is vaguely based on IGP2 simulation.py

class Sim4ADSimulation:

    def __init__(self,
                 scenario_map: Map,
                 dt: float = 0.1,
                 open_loop: bool = False,
                 episode: Episode = None):

        """ Initialise new simulation.

        Args:
            scenario_map: The current road layout.
            dt: Time difference between two time steps.
            open_loop: If true then no physical controller will be applied.
        """
        self.__scenario_map = scenario_map
        self.__dt = dt
        self.__fps = np.round(1 / dt)
        self.__open_loop = open_loop

        self.__time = 0
        self.__time_steps = 0
        self.__state = {}
        self.__agents = {}
        self.__actions = defaultdict(list)
        self.__episode = episode
        self.__agents_to_add = deepcopy(self.__episode.agents)  # Agents that have not been added to the simulation yet.
        self.__simulation_history = []

        # Dictionary (agent_id, DeathCause) of agents that have been removed from the simulation.
        self.__dead_agents = {}

        # History of observations, indexed by agent_id
        self.obs_history = defaultdict(list)

    def add_agent(self, new_agent: PolicyAgent):
        """ Add a new agent to the simulation.

        Args:
            new_agent: Agent to add.
        """
        if new_agent.agent_id in self.__agents \
                and self.__agents[new_agent.agent_id] is not None:
            raise ValueError(f"Agent with ID {new_agent.agent_id} already exists.")

        self.__agents[new_agent.agent_id] = new_agent
        self.__state[new_agent.agent_id] = new_agent.initial_state
        self.__agents_to_add.pop(new_agent.agent_id)

        logger.debug(f"Added Agent {new_agent.agent_id}")

    def _create_policy_agent(self, agent, policy: str = "BC"):

        center = np.array([float(agent.x_vec[0]), float(agent.y_vec[0])])
        heading = agent.psi_vec[0]
        lane = self.__scenario_map.best_lane_at(center, heading, max_distance=0.5)
        initial_state = State(time=0, position=center,
                              velocity=np.sqrt(float(agent.vx_vec[0]) ** 2 + float(agent.vy_vec[0]) ** 2),
                              acceleration=np.sqrt(float(agent.ax_vec[0]) ** 2 + float(agent.ay_vec[0]) ** 2),
                              heading=heading, lane=lane)

        if policy == "BC":
            # TODO: we predict (acceleration, steering angle) from the history of observations
            policy = BC(state_dim=34, action_dim=2)  # todo: make dynamic
            policy.load_policy()
        else:
            raise ValueError(f"Policy {policy} not found.")

        return PolicyAgent(agent_id=agent.UUID, initial_state=initial_state, policy=policy)

    def remove_agent(self, agent_id: int):
        """ Remove an agent from the simulation.

        Args:
            agent_id: Agent ID to remove.
        """
        self.__agents[agent_id].alive = False
        self.__agents.pop(agent_id)
        logger.debug(f"Removed Agent {agent_id}")

        # TODO: log the event for evaluation

    def reset(self):
        """ Remove all agents and reset internal state of simulation. """
        self.__time = 0
        self.__time_steps = 0
        self.__agents = {}
        self.__state = {}
        self.__dead_agents = {}
        self.__agents_to_add = deepcopy(self.__episode.agents)
        self.__simulation_history = []

    def step(self):
        """ Advance simulation by one time step. """
        logger.debug(f"Simulation step {self.__time_steps}")
        self.__update_vehicles()
        self.__take_actions()
        self.__time += self.__dt
        self.__time_steps += 1

    @property
    def time(self):
        """ Get the current time of the simulation. """
        return self.__time

    def __update_vehicles(self):
        """
        Spawn new vehicles in the simulation based on when they appear in the dataset.
        Remove the vehicles that are dead.
        """

        add_agents = {}
        for agent_id, agent in self.__agents_to_add.items():
            if (agent.time[0] - self.__time < self.__dt) and (agent.time[-1] - self.__time > 0):
                if agent_id not in self.__agents:
                    add_agents[agent_id] = agent

        for agent_id, agent in add_agents.items():
            self.add_agent(self._create_policy_agent(agent))

        for agent_id, death_cause in self.__dead_agents.items():
            self.remove_agent(agent_id)

            # TODO: log the event
            print(f"Agent {agent_id} has been removed from the simulation for {death_cause} at time {self.time}.")

        self.__dead_agents = {}

    def __take_actions(self):

        # A frame is a dictionary (agent_id, State)
        new_frame = {}

        for agent_id, agent in self.__agents.items():

            if agent is None:
                continue

            # TODO: for efficiency we could compute the observations of all agents at once
            obs = self._get_observation(agent=agent,
                                        state=self.__state[agent_id])  # Features of the agent's current state
            self.obs_history[agent_id].append(obs)
            action = agent.next_action(history=self.obs_history[agent_id])

            # Use the bicycle model to find where the agent will be at t+1
            new_state = self._next_state(agent, self.__state[agent_id], action=action)

            # TODO: given nearby agents, check if there is a collision
            collision = False  # TODO: check!
            done = agent.done()
            off_road = new_state.lane is None  # TODO: only compute this one in the file, not mutliple times across function # TODO: recycle this computation for later use

            if collision:
                self.__dead_agents[agent_id] = DeathCause.COLLISION
            elif off_road:
                self.__dead_agents[agent_id] = DeathCause.OFF_ROAD

                # Plot its trajectory
                # plot map
                plot_map(self.__scenario_map, markings=True, hide_road_bounds_in_junction=True)
                for state in agent.trajectory:
                    plt.plot(*state.position, marker="o")
                plt.show()

            elif done:
                self.__dead_agents[agent_id] = DeathCause.TIMEOUT  # todo: what if goal reached?

            dead = collision or done

            if not dead:
                # TODO: check
                # TODO: agent.trajectory.add_state(new_state, reload_path=False)
                agent.trajectory.append(new_state)
                self.__actions[agent_id].append(action)
                new_frame[agent_id] = new_state

            # TODO: agent.alive = len(self.__scenario_map.roads_at(new_state.position)) > 0

        new_frame["time"] = self.__time
        self.__simulation_history.append(new_frame)
        self.__state = new_frame

    def _next_state(self, agent: PolicyAgent, state: State, action: Action) -> State:
        """
        Compute the next state based on the current state and action using the bicycle model.

        Apply acceleration and steering according to the bicycle model centered at the center-of-gravity (i.e. cg)
        of the vehicle.

        Ref: https://dingyan89.medium.com/simple-understanding-of-kinematic-bicycle-model-81cac6420357

        :param state: The current State.
        :param action: Acceleration and steering action to execute
        :return: The next State.
        """

        acceleration = np.clip(action.acceleration, - agent.meta.max_acceleration, agent.meta.max_acceleration)

        velocity = state.velocity + acceleration * self.__dt
        velocity = max(0, velocity)

        beta = np.arctan(agent._l_r * np.tan(action.steer_angle) / agent.meta.wheelbase)
        d_position = np.array(
            [velocity * np.cos(beta + state.heading),
             velocity * np.sin(beta + state.heading)]
        )

        center = state.position + d_position * self.__dt
        d_theta = velocity * np.tan(action.steer_angle) * np.cos(beta) / agent.meta.wheelbase
        d_theta = np.clip(d_theta, - agent.meta.max_angular_vel, agent.meta.max_angular_vel)
        heading = (state.heading + d_theta * self.__dt + np.pi) % (2 * np.pi) - np.pi

        new_lane = self.__scenario_map.best_lane_at(center, heading, max_distance=0.5)

        return State(time=self.time, position=center, velocity=velocity, acceleration=acceleration, heading=heading,
                     lane=new_lane)

    def _get_observation(self, agent: PolicyAgent, state: State) -> Observation:
        """
        Get the current observation of the agent.

        :param agent: The agent.
        :return: The observation.
        """

        distance_left_lane_marking, distance_right_lane_marking = self._compute_distance_markings(state)
        nearby_agents = self._get_nearby_vehicles(agent=agent, state=state)

        front_ego = nearby_agents["center_front"]
        behind_ego = nearby_agents["center_behind"]
        left_front = nearby_agents["left_front"]
        left_behind = nearby_agents["left_behind"]
        right_front = nearby_agents["right_front"]
        right_behind = nearby_agents["right_behind"]

        observation = {
            "velocity": state.velocity,
            "heading": state.heading,
            "distance_left_lane_marking": distance_left_lane_marking,
            "distance_right_lane_marking": distance_right_lane_marking,
            "front_ego_rel_dx": front_ego["rel_dx"],
            "front_ego_rel_dy": front_ego["rel_dy"],
            "front_ego_v": front_ego["v"],
            "front_ego_a": front_ego["a"],
            "front_ego_heading": front_ego["heading"],
            "behind_ego_rel_dx": behind_ego["rel_dx"],
            "behind_ego_rel_dy": behind_ego["rel_dy"],
            "behind_ego_v": behind_ego["v"],
            "behind_ego_a": behind_ego["a"],
            "behind_ego_heading": behind_ego["heading"],
            "front_left_rel_dx": left_front["rel_dx"],
            "front_left_rel_dy": left_front["rel_dy"],
            "front_left_v": left_front["v"],
            "front_left_a": left_front["a"],
            "front_left_heading": left_front["heading"],
            "behind_left_rel_dx": left_behind["rel_dx"],
            "behind_left_rel_dy": left_behind["rel_dy"],
            "behind_left_v": left_behind["v"],
            "behind_left_a": left_behind["a"],
            "behind_left_heading": left_behind["heading"],
            "front_right_rel_dx": right_front["rel_dx"],
            "front_right_rel_dy": right_front["rel_dy"],
            "front_right_v": right_front["v"],
            "front_right_a": right_front["a"],
            "front_right_heading": right_front["heading"],
            "behind_right_rel_dx": right_behind["rel_dx"],
            "behind_right_rel_dy": right_behind["rel_dy"],
            "behind_right_v": right_behind["v"],
            "behind_right_a": right_behind["a"],
            "behind_right_heading": right_behind["heading"]
        }

        observation = Observation(state=observation)
        return observation.get_observation()

    def _compute_distance_markings(self, state: State) -> Tuple[float, float]:
        """
        Compute the distance to the left and right lane markings.

        :param state: The current state.
        :return: The distance to the left and right lane markings.
        """

        lane = state.lane
        ds_on_lane = lane.distance_at(state.position)

        if lane is None:
            # Plot map and position for debugging
            plot_map(self.__scenario_map, markings=True, hide_road_bounds_in_junction=True)
            plt.plot(*state.position, marker="o")
            plt.show()
            raise ValueError(f"No lane found at position {state.position} and heading {state.heading}.")

        # 1. find the slope of the line perpendicular to the lane through the agent

        position = Point(state.position)
        # Find the point on the boundary closest to the agent
        closest_point = lane.boundary.boundary.interpolate(lane.boundary.boundary.project(position))

        # We now want to find if the point is on the left or right side of the agent
        # We can do this by using the cross product of the vector v1 from (0,0,0) to the agent and
        # vector v2 from (0,0,0) to the closest point. If the cross product is positive, then the point is on the
        # right side of the agent, if it is negative, then it is on the left side.
        v1 = np.array([state.position[0], state.position[1], 0])
        v2 = np.array([closest_point.x, closest_point.y, 0])
        cross_product = np.cross(v1, v2)
        if cross_product[2] > 0:  # If the z-component is positive or negative
            # The point is on the right side of the agent=
            distance_right_lane_marking = position.distance(closest_point)
            distance_left_lane_marking = lane.get_width_at(ds_on_lane) - distance_right_lane_marking
        else:
            # The point is on the left side of the agent
            distance_left_lane_marking = position.distance(closest_point)
            distance_right_lane_marking = lane.get_width_at(ds_on_lane) - distance_left_lane_marking

        assert distance_left_lane_marking + distance_right_lane_marking - lane.get_width_at(ds_on_lane) < 1e-6

        return distance_left_lane_marking, distance_right_lane_marking

    def _find_perpendicular(self, lane: Lane, state: State, length=50, debug=False) -> LineString:

        # We need to take the tangent as we want the slope (ration dy/dx) and not the heading
        ds_on_lane = lane.distance_at(state.position)
        m = -1 / np.tan(lane.get_heading_at(ds_on_lane))

        # 2. find the equation of the line to the lane through the agent: y = m * (x - x0) + y0
        y = lambda x: m * (x - state.position[0]) + state.position[1]

        # 3. find the points lane's width away from the agent on the left and right using
        # https://math.stackexchange.com/questions/175896/finding-a-point-along-a-line-a-certain-distance-away-from-another-point

        if state.heading < 0:
            x_left = state.position[0] + length / np.sqrt(1 + m ** 2)
            x_right = state.position[0] - length / np.sqrt(1 + m ** 2)
        else:
            x_right = state.position[0] + length / np.sqrt(1 + m ** 2)
            x_left = state.position[0] - length / np.sqrt(1 + m ** 2)

        y_left = y(x_left)
        y_right = y(x_right)

        # Check if the slope is correct
        assert (y_right - y_left) / (x_right - x_left) - m < 1e-6

        perpendicular = LineString([(x_left, y_left), (x_right, y_right)])

        if debug is True:
            plot_map(self.__scenario_map, markings=True, hide_road_bounds_in_junction=True)
            plt.plot(*perpendicular.xy, color="blue", linewidth=2)
            plt.plot(*state.position, marker="o")

            # only show a 20x20 m window around the agent
            plt.xlim(state.position[0] - 10, state.position[0] + 10)
            plt.ylim(state.position[1] - 10, state.position[1] + 10)
            plt.show()

        return perpendicular

    def _get_nearby_vehicles(self, agent: PolicyAgent, state: State):

        """
        TODO: we assume that there is only one lane, and not consider that vehicle may be in different lane groups,
        e.g., if lane changes group in front and the agent in front is in that lane instead.
        """

        # 1. We need the id of the lane where the vehicle is on
        lane = state.lane

        # plot scenario and position of all vehicles # TODO
        if lane is None:
            plot_map(self.__scenario_map, markings=True, hide_road_bounds_in_junction=True)
            for state in agent.trajectory:
                plt.plot(*state.position, marker="o", color="blue")
            plt.show()
            raise ValueError(f"No lane found at position {state.position} and heading {state.heading}.")
        # plot_map(self.__scenario_map, markings=True, hide_road_bounds_in_junction=True)
        # for agent_id, agent in self.__agents.items():
        #     agent_state = agent.state
        #     plt.plot(*agent_state.position, marker="o", color="blue")

        # plt.show()

        # 2. We want the lanes to the left and right of the current one (as long as they have the same flow of motion)
        # TODO: in urban environments, is this an acceptable limitation, or should we also include vehicles from
        #   the other direction, as they may surpass, merge / cut in front of the vehicle?

        left_l, center_l, right_l = lane.traversable_neighbours(return_lfr_order=True)

        # 3. We want to further divide the lanes into two parts, the one in front and the one behind the vehicle.
        perpendicular = self._find_perpendicular(lane, state)

        # plot_map(self.__scenario_map, markings=True, hide_road_bounds_in_junction=True) # TODO
        # plt.plot(*perpendicular.xy, color="blue", linewidth=2)
        # plt.plot(*state.position, marker="o")

        nearby_lanes = {"left_front": None, "left_behind": None, "center_front": None,
                        "center_behind": None, "right_front": None, "right_behind": None}

        if left_l is not None:
            nearby_lanes["left_front"], nearby_lanes["left_behind"] = split(left_l.boundary, perpendicular).geoms
            # plt.plot(*nearby_lanes["left_front"].boundary.xy, color="orange")
            # plt.plot(*nearby_lanes["left_behind"].boundary.xy, color="green")
        if center_l is not None:
            nearby_lanes["center_front"], nearby_lanes["center_behind"] = split(center_l.boundary, perpendicular).geoms
            # plt.plot(*nearby_lanes["center_front"].boundary.xy, color="orange")
            # plt.plot(*nearby_lanes["center_behind"].boundary.xy, color="green")
        if right_l is not None:
            nearby_lanes["right_front"], nearby_lanes["right_behind"] = split(right_l.boundary, perpendicular).geoms
            # plt.plot(*nearby_lanes["right_front"].boundary.xy, color="blue")
            # plt.plot(*nearby_lanes["right_behind"].boundary.xy, color="yellow")

        # only show a 20x20 m window around the agent
        # plt.xlim(state.position[0] - 10, state.position[0] + 10)
        # plt.ylim(state.position[1] - 10, state.position[1] + 10)

        # plt.show()

        nearby_vehicles = defaultdict(None)

        # 4. For each of the (max 6) areas, we want to find the intersection of the vehicle with these areas.
        for area_name, area in nearby_lanes.items():

            closest_vehicle = None

            if area is not None:
                min_distance = float("inf")

                # Loop through all agents and check if they are in the area
                for agent_id, agent in self.__agents.items():
                    if agent_id == agent.agent_id:
                        continue

                    agent_state = agent.state
                    agent_position = Point(agent_state.position)
                    if area.contains(agent_position):
                        distance = Point(state.position).distance(agent_position)
                        if distance < min_distance:
                            min_distance = distance
                            closest_vehicle = agent

            nearby_vehicles[area_name] = self.__get_vehicle_features(closest_vehicle, state=state)

        return nearby_vehicles

    @staticmethod
    def __get_vehicle_features(vehicle: PolicyAgent, state: State):
        """
        :param vehicle: The vehicle to get the features from, w.r.t the ego agent.
        :param state: The current state of the ego agent.
        """
        features = defaultdict(lambda: MISSING_NEARBY_AGENT_VALUE)

        if vehicle is not None:
            features["rel_dx"] = vehicle.state.position[0] - state.position[0]
            features["rel_dy"] = vehicle.state.position[1] - state.position[1]
            features["v"] = vehicle.state.velocity
            features["a"] = vehicle.state.acceleration
            features["heading"] = vehicle.state.heading
        return features

    def replay_simulation(self):
        """
        Replay the simulation as a video using self.__simulation_history which is a list of frames, where each
        frame is a dictionary (agent_id, State). Also plot the map as background.
        """

        fig, ax = plt.subplots()
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)

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

                ax.plot(*state.position, marker="o", color=color)

                # Plot the ground truth position with a cross.
                original_agent = self.__episode.agents[agent_id]
                initial_time = original_agent.time[0]
                time_idx = int((time - initial_time) / self.__dt)
                ax.plot(original_agent.x_vec[time_idx], original_agent.y_vec[time_idx], marker="x", color=color)

            ax.set_title(f"Simulation Time: {time}")

        ani = FuncAnimation(fig, update, frames=self.__simulation_history)
        plt.show()


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
        0]  # TODO: maybe should take average across steps or something or hardcode it in the scneario config file

    sim = Sim4ADSimulation(scenario_map, episode=episode, dt=dt)
    sim.reset()

    simulation_length = 20  # seconds

    for _ in tqdm(range(int(np.floor(simulation_length / dt)))):
        sim.step()

    sim.replay_simulation()

    print("Simulation done!")
    raise NotImplementedError("Saving trajectories is not yet implemented.")

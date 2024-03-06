import logging
from collections import defaultdict
from typing import Dict, Any

import numpy as np

from sim4ad.data import Episode
from simulator.policy_agent import PolicyAgent
from simulator.simulator_util import DeathCause
from simulator.state_action import State
from simulator.simulator_util import PositionNearbyAgent as PNA

logger = logging.getLogger(__name__)


class EvaluationFeaturesExtractor:

    def __init__(self, sim_name: str):
        self.__sim_name = sim_name

        # A dictionary (indexed by agent id) of dictionaries (indexed by feature type) of features over time.
        self.__agents = defaultdict(lambda: defaultdict(list))

    def save_trajectory(self, agent: PolicyAgent, death_cause: DeathCause):
        """
        Save the trajectory of the agent in the simulation for evaluation.
        :param agent: agent
        :param death_cause: cause of death
        """

        states = agent.state_trajectory
        states = states[:-1]  # Remove the last state, as it is the one where the agent died.
        obs = agent.observation_trajectory
        actions = agent.action_trajectory
        agent.alive = False

        assert len(states) == len(obs) == len(actions)

        self.agents[agent.agent_id]["death_cause"] = death_cause.value
        self.agents[agent.agent_id]["states"] = states
        self.agents[agent.agent_id]["observations"] = obs
        self.agents[agent.agent_id]["actions"] = actions

        # compute ttcs and tths
        for i in range(len(states)):  # -1 because the last state is the one where the agent died.
            self._compute_ttc_tth(agent, states[i], agent.nearby_vehicles[i])

        self.agents[agent.agent_id]["nearby_vehicles"] = agent.nearby_vehicles
        self.agents[agent.agent_id]["right_marking_distance"] = agent.distance_right_lane_marking
        self.agents[agent.agent_id]["left_marking_distance"] = agent.distance_left_lane_marking
        self.agents[agent.agent_id]["time"] = [state.time for state in states]

    def _compute_ttc_tth(self, agent: PolicyAgent, state: State, nearby_vehicles: Dict[PNA, Dict[str, Any]]):
        """
        Compute the time-to-collision (TTC) for the agent.
        Computed as outlined in https://openautomatumdronedata.readthedocs.io/en/latest/readme_include.html

        :param agent: The agent w.r.t which we are computing the TTC/TTH.
        :param state: The state of the agent.
        :param nearby_vehicles: The agents nearby the agent. Each agent then has a dictionary of features.
        """

        # Dictionary of TTCs and TTHs indexed by the position of the nearby agent (e.g., "center_front", "right_front").
        ttcs = {}
        tths = {}

        for position, nearby_agent in nearby_vehicles.items():

            # TTC and TTH are only computed for the agents in front
            if nearby_agent is None or position not in [PNA.CENTER_IN_FRONT, PNA.RIGHT_IN_FRONT, PNA.LEFT_IN_FRONT]:
                ttc = None
                tth = None
            else:
                d = (state.position.distance(nearby_agent["position"]) - agent.meta.length / 2 -
                     nearby_agent["metadata"].length / 2)
                v_ego = state.speed
                v_other = nearby_agent["speed"]

                if v_other < v_ego:
                    ttc = d / (v_ego - v_other)
                else:
                    # A collision is impossible if the other agent is faster than the ego
                    ttc = -1

                tth = d / v_ego

            ttcs[position] = ttc
            tths[position] = tth

        self.__agents[agent.agent_id]["TTC"].append(ttcs)
        self.__agents[agent.agent_id]["TTH"].append(tths)

    def rmse_speed(self, real_speeds: Dict[str, Dict[str, np.ndarray]]):
        """
        Compute the (average) root-mean-square error (RMSE) of the speed of ALL agents, compared to the dataset.

        :param real_speeds: The real speeds of the agents in the dataset. Dict[agent_id, speeds] where speeds is a
        dictionary of speeds, indexed as (vx_s, vy_s), and the values are numpy arrays of the speeds.
        """

        rmse = 0

        for agent_id, features in self.__agents.items():
            real_vel_x = real_speeds[agent_id]["vx_s"]
            real_vel_y = real_speeds[agent_id]["vy_s"]
            real_speed = np.sqrt(np.array(real_vel_x) ** 2 + np.array(real_vel_y) ** 2)
            simulated_speed = np.array([state.speed for state in features["states"]])

            if len(real_speed) != len(simulated_speed):
                if features["death_cause"] == DeathCause.TIMEOUT.value:
                    # If the agent timed out, the length of the real trajectory is longer than the simulated one.
                    real_speed = real_speed[:len(simulated_speed)]
                else:
                    logger.error(f"Real and simulated trajectories have different lengths for agent {agent_id}.")
                    # TODO: could add discounting for the missing steps
                    raise ValueError("Real and simulated trajectories have different lengths.")

            rmse += np.sqrt(np.mean((real_speed - simulated_speed) ** 2))

        return rmse / len(self.__agents)

    def rmse_position(self, real_position: Dict[str, Dict[str, np.ndarray]]):
        """
        Compute the (average) root-mean-square error (RMSE) of the position of ALL agents, compared to the dataset.
        """

        rmse = 0

        for agent_id, features in self.__agents.items():
            real_x = real_position[agent_id]["x_s"]
            real_y = real_position[agent_id]["y_s"]
            simulated_x = np.array([state.position.x for state in features["states"]])
            simulated_y = np.array([state.position.y for state in features["states"]])

            if len(real_x) != len(simulated_x):
                if features["death_cause"] == DeathCause.TIMEOUT.value:
                    # If the agent timed out, the length of the real trajectory is longer than the simulated one.
                    real_x = real_x[:len(simulated_x)]
                    real_y = real_y[:len(simulated_y)]
                else:
                    logger.error(f"Real and simulated trajectories have different lengths for agent {agent_id}.")
                    # TODO: could add discounting for the missing steps
                    raise ValueError("Real and simulated trajectories have different lengths.")

            rmse_x = np.sqrt(np.mean((real_x - simulated_x) ** 2))
            rmse_y = np.sqrt(np.mean((real_y - simulated_y) ** 2))

            rmse += rmse_y + rmse_x

        return rmse / len(self.__agents)

    def compute_out_of_road_rate(self):
        """
        Compute the rate of agents that are out of the road. I.e., those who has OFF_ROAD as death cause.

        :return fraction of agents that are out of the road.
        """

        out_of_road_agents = [agent for agent in self.__agents.values() if agent["death_cause"] == DeathCause.OFF_ROAD.value]

        return len(out_of_road_agents) / len(self.__agents)

    def compute_collision_rate(self):
        """
        Compute the rate of agents that have collided. I.e., those who has COLLISION as death cause.
        :return: fraction of agents that have collided.
        """

        collision_agents = [agent for agent in self.__agents.values() if agent["death_cause"] == DeathCause.COLLISION.value]

        return len(collision_agents) / len(self.__agents)

    def compute_realism_scores(self, ground_truth: Episode = None):
        # TODO: Implement this method.

        if ground_truth is not None:
            # TODO: Implement this method.
            rmse_speed = self.rmse_speed(real_speeds=ground_truth.agents)
            rmse_position = self.rmse_position()

        collision_rate = self.compute_collision_rate()
        out_of_road_rate = self.compute_out_of_road_rate()

        raise NotImplementedError("This method is not implemented yet.")


    @property
    def agents(self):
        return self.__agents

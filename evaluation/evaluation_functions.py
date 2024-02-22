import logging
from collections import defaultdict
from typing import Dict, Any

from simulator.policy_agent import PolicyAgent
from simulator.simulator_util import DeathCause
from simulator.state_action import State
from simulator.simulator_util import PositionNearbyAgent as PNA

logger = logging.getLogger(__name__)


class EvaluationFeaturesExtractor:

    def __init__(self, sim_name:str):
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
        obs = agent.observation_trajectory
        actions = agent.action_trajectory
        agent.alive = False

        assert (len(states) - 1) == len(obs) == len(actions)  # There is an extra state at the end

        logger.error("This method is not implemented yet.")
        # TODO

    def compute_features(self, agent: PolicyAgent, state: State, nearby_vehicles: Dict[PNA, Dict[str, Any]],
                         right_marking_distance: float, left_marking_distance: float):
        """
        Compute the evaluation features for the agents in the simulation.

        :param agent: The agent for which we are computing the features.
        :param state: The state of the agent.
        :param nearby_vehicles: The agents nearby the agent. Each agent then has a dictionary of features.
        :param right_marking_distance: The distance to the right marking.
        :param left_marking_distance: The distance to the left marking.
        """

        self.agents[agent.agent_id]["death_cause"] = None
        self.agents[agent.agent_id]["time"].append(state.time)
        self.agents[agent.agent_id]["nearby_vehicles"].append(nearby_vehicles)
        self.agents[agent.agent_id]["right_marking_distance"].append(right_marking_distance)
        self.agents[agent.agent_id]["left_marking_distance"].append(left_marking_distance)
        self._compute_ttc_tth(agent, state, nearby_vehicles)

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
                d = (state.position.distance(nearby_agent.state.position) - agent.meta.length/2 -
                     nearby_agent.meta.length/2)
                v_ego = state.velocity
                v_other = nearby_agent.state.velocity

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

    # TODO: for each feature + R/L marking distance, I could test it with real data from the dataset!

    def compute_realism_scores(self):
        # TODO: Implement this method.
        raise NotImplementedError("This method is not implemented yet.")

    @property
    def agents(self):
        return self.__agents

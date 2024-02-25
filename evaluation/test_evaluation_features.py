"""
Given the dataset, sample random agents and states and compare the computed features with the features in the dataset.
"""

import unittest
from collections import defaultdict

import logging
import numpy as np
from tqdm import tqdm

from sim4ad.data import DatasetDataLoader
from sim4ad.opendrive import Map
from simulator.lightweight_simulator import Sim4ADSimulation
from simulator.simulator_util import DeathCause

logger = logging.getLogger(__name__)


class TestEvaluationFeatures(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestEvaluationFeatures, self).__init__(*args, **kwargs)
        self.__agents = defaultdict(lambda: defaultdict(list))
        self.__sim_name = "test_sim"
        self.__simulation_agent_features = None

    def _setup_simulation(self):

        # TODO: could change the map
        # TODO: maybe make a function to load a map since we reuse it in the simulator too
        scenario_map = Map.parse_from_opendrive(
            "scenarios/data/automatum/hw-a9-appershofen-001-d8087340-8287-46b6-9612-869b09e68448/staticWorld.xodr")

        data_loader = DatasetDataLoader(f"scenarios/configs/appershofen.json")
        data_loader.load()

        episodes = data_loader.scenario.episodes

        # TODO: loop over episodes
        self.episode = episodes[0]

        agent = list(self.episode.agents.values())[0]

        # Take the time difference between steps to be the gap in the dataset
        dt = agent.delta_t
        sim = Sim4ADSimulation(scenario_map, episode=self.episode, dt=dt, policy_type="follow_dataset")

        simulation_length = 150  # seconds

        for _ in tqdm(range(int(np.floor(simulation_length / sim.dt)))):
            sim.step()

        self.__simulation_evaluator = sim.evaluator
        self.__simulation_agent_features = sim.evaluator.agents

    def test_time_steps_equal(self):

        if self.__simulation_agent_features is None:
            self._setup_simulation()

        # for each agent, compare the tth and ttc features with the dataset
        for agent_id, features in self.__simulation_agent_features.items():

            # assert the times are approximately the same across the dataset and the simulation
            try:
                assert np.allclose(features["time"], self.episode.agents[agent_id].time)
            except ValueError:
                if features["death_cause"] is DeathCause.TIMEOUT.value:
                    # The agent is alive but has been cut off by the simulation
                    logger.warning(f"Agent {agent_id} is alive but has been cut off by the simulation.")

                    # Check the times match up to the last time the agent was alive
                    last_time_alive_idx = len(features["time"])
                    assert np.allclose(features["time"], self.episode.agents[agent_id].time[:last_time_alive_idx])
                else:
                    raise ValueError(f"Agent {agent_id} is dead but the simulation has not killed it.")

    def test_nearby_vehicles_match(self):
        """
        Compare the id of the nearby vehicles in the dataset with the id of the nearby vehicles in the simulation.
        """

        if self.__simulation_agent_features is None:
            self._setup_simulation()

        for agent_id, features in self.__simulation_agent_features.items():
            real_nearby_vehicles = self.episode.agents[agent_id].object_relation_dict_list
            simulated_nearby_vehicles = features["nearby_vehicles"]

            miscounts = []

            for step_idx in range(len(simulated_nearby_vehicles)):
                for position in simulated_nearby_vehicles[step_idx]:
                    if simulated_nearby_vehicles[step_idx][position] is None:
                        if real_nearby_vehicles[step_idx][position.value] is not None:
                            miscounts.append(
                                f"Real nearby vehicles: {real_nearby_vehicles[step_idx][position.value]}, Simulated nearby vehicles: {simulated_nearby_vehicles[step_idx][position]}, at step {step_idx} for agent {agent_id}, position {position}")
                    else:
                        if real_nearby_vehicles[step_idx][position.value] != \
                                simulated_nearby_vehicles[step_idx][position]["agent_id"]:
                            miscounts.append(
                                f"Real nearby vehicles: {real_nearby_vehicles[step_idx][position.value]}, Simulated nearby vehicles: {simulated_nearby_vehicles[step_idx][position]}, at step {step_idx} for agent {agent_id}, position {position}")

            if len(miscounts) > 0:  # TODO: should miscounts be across the entire dataset or just for one agent?
                logger.warning(f"Found {len(miscounts)} miscounts: {miscounts}")

            # TODO: Only raise a concern if the miscounts are more than 5% of the dataset
            assert len(miscounts) < (0.05 * len(real_nearby_vehicles))

    def test_TTC_TTH_match(self):
        """
        Test the computation of the time-to-collision (TTC) and time-to-heading (TTH) features.
        """

        if self.__simulation_agent_features is None:
            self._setup_simulation()

        for agent_id, features in self.__simulation_agent_features.items():
            real_ttc = self.episode.agents[agent_id].ttc_dict_vec
            real_tth = self.episode.agents[agent_id].tth_dict_vec

            simulated_ttc = features["TTC"]
            simulated_tth = features["TTH"]

            miscounts = []

            # assert the times are approximately the same across the dataset and the simulation
            for step_idx in range(len(simulated_ttc)):
                # As ttc and tth is only computed for the agents in front, we only need to check the positions in front
                for position in simulated_ttc[step_idx]:

                    if simulated_ttc[step_idx][position] is None or real_ttc[step_idx][position.value] is None:
                        if not (simulated_ttc[step_idx][position] is None and real_ttc[step_idx][
                            position.value] is None):
                            miscounts.append(
                                f"Real TTC: {real_ttc[step_idx][position.value]}, Simulated TTC: {simulated_ttc[step_idx][position]}, at step {step_idx} for agent {agent_id}, position {position}")
                    else:
                        if real_ttc[step_idx][position.value] - simulated_ttc[step_idx][position] > 0.001:
                            miscounts.append(
                                f"Real TTC: {real_ttc[step_idx][position.value]}, Simulated TTC: {simulated_ttc[step_idx][position]}, at step {step_idx} for agent {agent_id}, position {position}")
                    if simulated_tth[step_idx][position] is None or real_tth[step_idx][position.value] is None:
                        if not (simulated_tth[step_idx][position] is None and real_tth[step_idx][
                            position.value] is None):
                            miscounts.append(
                                f"Real TTH: {real_tth[step_idx][position.value]}, Simulated TTH: {simulated_tth[step_idx][position]}, at step {step_idx} for agent {agent_id}, position {position}")

                    else:
                        if real_tth[step_idx][position.value] - simulated_tth[step_idx][position] > 0.001:
                            miscounts.append(
                                f"Real TTH: {real_tth[step_idx][position.value]}, Simulated TTH: {simulated_tth[step_idx][position]}, at step {step_idx} for agent {agent_id}, position {position}")

            if len(miscounts) > 0:
                logger.warning(f"Found {len(miscounts)} miscounts: {miscounts}")

            assert len(miscounts) < (0.05 * len(real_ttc))

    def test_right_left_marking_distance(self):
        """
        Test the computation of the right and left marking distance features.
        """

        if self.__simulation_agent_features is None:
            self._setup_simulation()

        for agent_id, features in self.__simulation_agent_features.items():
            real_right_marking_distance = self.episode.agents[agent_id].distance_right_lane_marking
            real_left_marking_distance = self.episode.agents[agent_id].distance_left_lane_marking

            simulated_right_marking_distance = features["right_marking_distance"]
            simulated_left_marking_distance = features["left_marking_distance"]

            miscounts = []

            # assert the times are approximately the same across the dataset and the simulation
            for step_idx in range(len(simulated_right_marking_distance)):
                if real_right_marking_distance[step_idx] - simulated_right_marking_distance[step_idx] > 0.001:
                    miscounts.append(
                        f"Real right marking distance: {real_right_marking_distance[step_idx]}, Simulated right marking distance: {simulated_right_marking_distance[step_idx]}, at step {step_idx} for agent {agent_id}")
                if real_left_marking_distance[step_idx] - simulated_left_marking_distance[step_idx] > 0.001:
                    miscounts.append(
                        f"Real left marking distance: {real_left_marking_distance[step_idx]}, Simulated left marking distance: {simulated_left_marking_distance[step_idx]}, at step {step_idx} for agent {agent_id}")

            if len(miscounts) > 0:
                logger.warning(f"Found {len(miscounts)} miscounts: {miscounts}")

            assert len(miscounts) < (0.05 * len(real_right_marking_distance))

    def test_speed_rmse(self):
        """Check that if we replay the dataset, the speed RMSE is 0."""

        if self.__simulation_agent_features is None:
            self._setup_simulation()

        assert self.__simulation_evaluator.rmse_speed() < 0.001

    def test_position_rmse(self):
        """Check that if we replay the dataset, the position RMSE is 0."""

        if self.__simulation_agent_features is None:
            self._setup_simulation()

        assert self.__simulation_evaluator.rmse_position() < 0.001


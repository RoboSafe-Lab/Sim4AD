"""
Given the dataset, sample random agents and states and compare the computed features with the features in the dataset.
"""

import unittest
from collections import defaultdict

import logging
import numpy as np
from openautomatumdronedata.dataset import droneDataset
from tqdm import tqdm

from sim4ad.data import DatasetDataLoader, ScenarioConfig, DatasetScenario
from sim4ad.opendrive import Map
from sim4ad.path_utils import get_path_to_automatum_scenario, get_config_path, get_agent_id_combined
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

        ep_name = ["hw-a9-appershofen-001-d8087340-8287-46b6-9612-869b09e68448",
                   "hw-a9-appershofen-002-2234a9ae-2de1-4ad4-9f43-65c2be9696d6"]

        self.gt_agents_data = get_ground_truth_data(ep_name)
        sim = Sim4ADSimulation(episode_name=ep_name, policy_type="follow_dataset", spawn_method="dataset_all")
        # TODO test this as well
        sim_micro_original = Sim4ADSimulation(episode_name=ep_name, policy_type="follow_dataset", spawn_method="dataset_one")

        # Simulation where all agents are spawned in random positions
        sim_random = Sim4ADSimulation(episode_name=ep_name, policy_type="bc-all-obs-1.5_pi", spawn_method="random")

        simulation_length = 50  # seconds

        for _ in tqdm(range(int(np.floor(simulation_length / sim.dt)))):
            sim.step()
            sim_micro_original.step()
            sim_random.step()

        sim.kill_all_agents()
        sim_micro_original.kill_all_agents()
        sim_random.kill_all_agents()

        self.__simulation_evaluator = sim.evaluator
        self.__simulation_agent_features = sim.evaluator.agents

        self.__micro_simulation_evaluator = sim_micro_original.evaluator
        self.__micro_simulation_agent_features = sim_micro_original.evaluator.agents

        self.__random_simulation_evaluator = sim_random.evaluator
        self.__random_simulation_agent_features = sim_random.evaluator.agents

    def test_time_steps_equal(self):

        if self.__simulation_agent_features is None:
            self._setup_simulation()

        for i, simulation_method in enumerate([self.__simulation_agent_features, self.__micro_simulation_agent_features]):
            print(f"Testing simulation method: {i}")
            for agent_id, features in simulation_method.items():

                # assert the times are approximately the same across the dataset and the simulation
                try:
                    assert np.allclose(features["time"], self.gt_agents_data[agent_id].time)
                except ValueError:
                    if features["death_cause"] is DeathCause.TIMEOUT.value:
                        # The agent is alive but has been cut off by the simulation
                        logger.warning(f"Agent {agent_id} is alive but has been cut off by the simulation.")

                        # Check the times match up to the last time the agent was alive
                        last_time_alive_idx = len(features["time"])
                        assert np.allclose(features["time"], self.gt_agents_data[agent_id].time[:last_time_alive_idx])
                    else:
                        raise ValueError(f"Agent {agent_id} is dead but the simulation has not killed it.")

    def test_nearby_vehicles_match(self):
        """
        Compare the id of the nearby vehicles in the dataset with the id of the nearby vehicles in the simulation.
        """

        if self.__simulation_agent_features is None:
            self._setup_simulation()

        for i, simulation_method in enumerate([self.__simulation_agent_features, self.__micro_simulation_agent_features]):
            print(f"Testing simulation method: {i}")
            for agent_id, features in simulation_method.items():
                real_nearby_vehicles = self.gt_agents_data[agent_id].object_relation_dict_list
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

        for i, simulation_method in enumerate([self.__simulation_agent_features, self.__micro_simulation_agent_features]):
            print(f"Testing simulation method: {i}")
            for agent_id, features in simulation_method.items():
                real_ttc = self.gt_agents_data[agent_id].ttc_dict_vec
                real_tth = self.gt_agents_data[agent_id].tth_dict_vec

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
                            if real_ttc[step_idx][position.value] - simulated_ttc[step_idx][position] > 0.01:
                                miscounts.append(
                                    f"Real TTC: {real_ttc[step_idx][position.value]}, Simulated TTC: {simulated_ttc[step_idx][position]}, at step {step_idx} for agent {agent_id}, position {position}")
                        if simulated_tth[step_idx][position] is None or real_tth[step_idx][position.value] is None:
                            if not (simulated_tth[step_idx][position] is None and real_tth[step_idx][
                                position.value] is None):
                                miscounts.append(
                                    f"Real TTH: {real_tth[step_idx][position.value]}, Simulated TTH: {simulated_tth[step_idx][position]}, at step {step_idx} for agent {agent_id}, position {position}")

                        else:
                            if real_tth[step_idx][position.value] - simulated_tth[step_idx][position] > 0.01:
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


        for i, simulation_method in enumerate([self.__simulation_agent_features, self.__micro_simulation_agent_features]):
            print(f"Testing simulation method: {i}")
            for agent_id, features in simulation_method.items():

                real_right_marking_distance = self.gt_agents_data[agent_id].distance_right_lane_marking
                real_left_marking_distance = self.gt_agents_data[agent_id].distance_left_lane_marking

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

        real_speeds = {}
        for agent_id, agent in self.gt_agents_data.items():
            real_speeds[agent_id] = {"vx_s": agent.vx_vec, "vy_s": agent.vy_vec}

        assert self.__simulation_evaluator.rmse_speed(real_speeds=real_speeds) < 0.001

    def test_position_rmse(self):
        """Check that if we replay the dataset, the position RMSE is 0."""

        if self.__simulation_agent_features is None:
            self._setup_simulation()

        real_position = {}
        for agent_id, agent in self.gt_agents_data.items():
            real_position[agent_id] = {"x_s": agent.x_vec, "y_s": agent.y_vec}

        assert self.__simulation_evaluator.rmse_position(real_position=real_position) < 0.001

    def test_collision_rate(self):
        """Check that if we replay the dataset, the collision rate is 0."""

        if self.__simulation_agent_features is None:
            self._setup_simulation()

        assert self.__simulation_evaluator.compute_collision_rate() == 0.

        # Check that in the random simulation, it is not 0
        # assert self.__random_simulation_evaluator.compute_collision_rate() > 0.

    def test_scores(self):
        """
        Check that the similarity scores are within a certain range.
        """

        if self.__simulation_agent_features is None:
            self._setup_simulation()

        _, scores = self.__simulation_evaluator.compute_similarity_scores(gt_agents_data=self.gt_agents_data)
        _, scores_micro = self.__micro_simulation_evaluator.compute_similarity_scores(gt_agents_data=self.gt_agents_data)

        for i, score in scores.items():
            assert (1 -  np.round(score, 1)) == 0., f"Score: {score}, i: {i}"

        for i, score in scores_micro.items():
            assert (1 -  np.round(score, 1)) == 0. , f"Score: {score}, i: {i}"


def get_ground_truth_data(episode_names):
    """
    Given alist of episodes, get the value for each agent by combining the episdode_id and the agent_id

    """
    gt_agents_data = {}

    for episode_name in episode_names:
        path_to_dataset_folder = get_path_to_automatum_scenario(episode_name)
        dataset = droneDataset(path_to_dataset_folder)
        dyn_world = dataset.dynWorld
        # dt = dyn_world.delta_t

        scenario_name = episode_name.split("-")[2]
        config = ScenarioConfig.load(get_config_path(scenario_name))
        data_loader = DatasetScenario(config)
        episode = data_loader.load_episode(episode_id=episode_name)

        for agent_id, agent in episode.agents.items():
            gt_agents_data[get_agent_id_combined(episode_name=episode_name, agent_id=agent_id)] = agent
    return gt_agents_data
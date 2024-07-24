import logging
from collections import defaultdict
from typing import Dict, Any, List
import numpy as np
from scipy.spatial import distance
from stable_baselines3 import SAC
from openautomatumdronedata.dataset import droneDataset

from sim4ad.data import ScenarioConfig, DatasetScenario
from sim4ad.opendrive import plot_map, Map
from sim4ad.path_utils import get_agent_id_combined, get_path_to_automatum_scenario, get_config_path
from simulator.policy_agent import PolicyAgent
from simulator.simulator_util import DeathCause
from simulator.state_action import State
from simulator.simulator_util import PositionNearbyAgent as PNA
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class EvaluationFeaturesExtractor:

    def __init__(self, sim_name: str, episode_names: List[str] = None):
        self.__sim_name = sim_name

        # A dictionary (indexed by agent id) of dictionaries (indexed by feature type) of features over time.
        self.__agents = defaultdict(lambda: defaultdict(list))
        self.__computed_metrics = defaultdict(dict)  # Store them as [metric][agent_id] = value
        if episode_names is not None:
            gt_agents_data = self.get_ground_truth_data(episode_names)
            self.__real_position, self.__real_speeds, self.__real_ttcs, self.__real_tths = self.get_ground_truth_info(
                gt_agents_data)

    @staticmethod
    def get_ground_truth_data(episode_names):
        """ Given a list of episodes, get the value for each agent by combining the episode_id and the agent_id """
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

    @staticmethod
    def get_ground_truth_info(gt_agents_data):
        """Get the real position, speeds, ttcs and tths"""
        # Fidelity Metrics
        real_position = {}
        real_speeds = {}
        real_ttcs = {}
        real_tths = {}

        # get the corresponding_gt data for our agents only
        # gt_agents_data = {agent_id: gt_agents_data[agent_id] for agent_id in self.__agents.keys()}

        for agent_id, agent in gt_agents_data.items():
            real_position[agent_id] = {"x_s": agent.x_vec, "y_s": agent.y_vec}
            real_speeds[agent_id] = {"vx_s": agent.vx_vec, "vy_s": agent.vy_vec}
            real_ttcs[agent_id] = agent.ttc_dict_vec
            real_tths[agent_id] = agent.tth_dict_vec

        return real_position, real_speeds, real_ttcs, real_tths

    def save_trajectory(self, agent: PolicyAgent, death_cause: DeathCause, episode_id: str):
        """
        Save the trajectory of the agent in the simulation for evaluation.
        :param agent: agent
        :param death_cause: cause of death
        :param episode_id:
        """

        states = agent.state_trajectory
        states = states[:-1]  # Remove the last state, as it is the one where the agent died.
        obs = agent.observation_trajectory
        actions = agent.action_trajectory

        if death_cause == DeathCause.TIMEOUT:
            # The agent didn't have a chance to complete the episode before the simulator ended. We therefore have
            # one more observation than action and state.
            obs = obs[:-1]
        # assert len(states) == len(obs) == len(actions)

        aid = get_agent_id_combined(episode_id, agent.agent_id)

        self.agents[aid]["death_cause"] = death_cause.value
        self.agents[aid]["states"] = states
        self.agents[aid]["observations"] = obs
        # self.agents[agent.agent_id]["actions"] = actions

        # compute ttcs and tths
        for i in range(len(states)):
            self.compute_ttc_tth(agent, states[i], agent.nearby_vehicles[i], episode_id, add=True)

        self.agents[aid]["nearby_vehicles"] = agent.nearby_vehicles
        self.agents[aid]["midline_distance"] = agent.distance_midline
        self.agents[aid]["time"] = [state.time for state in states]
        if isinstance(agent.policy, str):
            self.agents[aid]["policy_type"] = agent.policy
        elif isinstance(agent.policy, SAC):
            self.agents[aid]["policy_type"] = "sac"
        else:
            self.agents[aid]["policy_type"] = agent.policy.name

        self.agents[aid]["interference"] = agent.interferences

    def get_picklable_agents(self):
        """
        Return the agents in a format that can be pickled.
        """
        return dict(self.__agents)

    def compute_ttc_tth(self, agent: PolicyAgent, state: State, nearby_vehicles: Dict[PNA, Dict[str, Any]], episode_id,
                        add=True):
        """
        Compute the time-to-collision (TTC) and TTH for the agent.

        :param agent: The agent w.r.t which we are computing the TTC/TTH.
        :param state: The state of the agent.
        :param nearby_vehicles: The agents nearby the agent. Each agent then has a dictionary of features.
        :param episode_id:
        :param add:
        """

        # Dictionary of TTCs and TTHs indexed by the position of the nearby agent (e.g., "center_front", "right_front").
        ttc, tth = None, None

        for position, nearby_agent in nearby_vehicles.items():

            # TTC and TTH are only computed for the agents in front
            if position == PNA.CENTER_IN_FRONT and nearby_agent is not None:
                d = (state.position.distance(nearby_agent["position"]) - agent.meta.length / 2 -
                     nearby_agent["metadata"].length / 2)
                v_ego = abs(state.speed)
                v_other = abs(nearby_agent["speed"])

                ttc = d / (v_ego - v_other)
                # keep the same value as the dataset
                if ttc < 0:
                    ttc = -1
                tth = d / v_ego

        if add:
            aid = get_agent_id_combined(episode_id, agent.agent_id)
            self.__agents[aid]["TTC"].append(ttc)
            self.__agents[aid]["TTH"].append(tth)
        else:
            return ttc, tth

    def plot_criticality_distribution(self):
        """plot criticality distribution for diversity analysis"""
        all_simulated_ittcs, all_real_ittcs, all_simulated_tths, all_real_tths = self.compute_all_ittc_tth()

        self.plot_distogram('iTTC', real_data=all_real_ittcs, simulated_data=all_simulated_ittcs)
        self.plot_distogram('THW', real_data=all_real_tths, simulated_data=all_simulated_tths)

        plt.show()

    def get_simulated_real_speeds(self):

        real_speeds = {}
        simulated_speeds = {}

        for agent_id, features in self.__agents.items():
            real_vel_x = self.__real_speeds[agent_id]["vx_s"]
            real_vel_y = self.__real_speeds[agent_id]["vy_s"]
            real_speed = np.sqrt(np.array(real_vel_x) ** 2 + np.array(real_vel_y) ** 2)
            simulated_speed = np.array([state.speed for state in features["states"]])

            if len(real_speed) != len(simulated_speed):
                if len(real_speed) > len(simulated_speed):
                    real_speed = real_speed[:len(simulated_speed)]
                else:
                    # cut the simulated speed to match the real speed
                    simulated_speed = simulated_speed[:len(real_speed)]

            real_speeds[agent_id] = real_speed
            simulated_speeds[agent_id] = simulated_speed

        return real_speeds, simulated_speeds

    def plot_speed_distribution(self):
        """Plot speed distributions for human likeness analysis"""
        real_speeds, simulated_speeds = self.get_simulated_real_speeds()
        self.plot_distogram('Speed (m/s)', real_data=real_speeds, simulated_data=simulated_speeds)

        plt.show()

    def get_real_simulated_xy(self, real_position: Dict[str, Dict[str, np.ndarray]]):
        """
        Get the real and simulated x, y positions and speeds for a given agent.
        """
        real_xs = {}
        real_ys = {}
        simulated_xs = {}
        simulated_ys = {}

        for agent_id, features in self.__agents.items():
            real_x = real_position[agent_id]["x_s"]
            real_y = real_position[agent_id]["y_s"]
            simulated_x = np.array([state.position.x for state in features["states"]])
            simulated_y = np.array([state.position.y for state in features["states"]])

            if len(real_x) != len(simulated_x):
                if len(real_x) > len(simulated_x):
                    real_x = real_x[:len(simulated_x)]
                    real_y = real_y[:len(simulated_y)]
                else:
                    # cut the simulated speed to match the real speed
                    simulated_x = simulated_x[:len(real_x)]
                    simulated_y = simulated_y[:len(real_y)]

            real_xs[agent_id] = real_x
            real_ys[agent_id] = real_y
            simulated_xs[agent_id] = simulated_x
            simulated_ys[agent_id] = simulated_y

        return real_xs, real_ys, simulated_xs, simulated_ys

    def compute_all_ittc_tth(self):
        """
        Compute the inverse TTC (iTTC) and TTH distributions of the agents.
        """
        all_simulated_ittcs = []
        all_simulated_tths = []
        all_real_ittcs = []
        all_real_tths = []

        for agent_id, features in self.__agents.items():
            for v in features["TTC"]:
                if v is not None:
                    ittc = 1.0 / v
                    if ittc > 1: # > 1 means critical
                        ittc = 1.0
                    elif ittc < 0: # < 0 means uncritical
                        ittc = 0
                    all_simulated_ittcs.append(ittc)
            all_simulated_tths.extend(v for v in features["TTH"] if v is not None)
        for features in self.__real_ttcs.values():
            for v in features:
                if v['front_ego'] is not None:
                    ittc = 1.0 / v['front_ego']
                    if ittc > 1:
                        ittc = 1.0
                    elif ittc < 0:
                        ittc = 0
                    all_real_ittcs.append(ittc)
        for features in self.__real_tths.values():
            all_real_tths.extend(v['front_ego'] for v in features if v['front_ego'] is not None)

        return all_simulated_ittcs, all_real_ittcs, all_simulated_tths, all_real_tths

    @staticmethod
    def compute_jsd(simulated_values, real_values):
        """
        Get two arrays with simulated and real values. Create a histogram of the values and compute the Jensen-Shannon
        divergence between the two distributions.
        """

        # Compute the histograms of the distributions
        nr_bins = 10
        simulated_hist, _ = np.histogram(simulated_values, bins=nr_bins, density=True)
        real_hist, _ = np.histogram(real_values, bins=nr_bins, density=True)

        # Compute the JSD between the simulated and real distributions
        # Add a small value to avoid division by zero
        simulated_hist = simulated_hist + 1e-6
        real_hist = real_hist + 1e-6

        simulated_hist = simulated_hist / np.sum(simulated_hist)
        real_hist = real_hist / np.sum(real_hist)

        jsd = distance.jensenshannon(simulated_hist, real_hist)
        if np.isnan(jsd):
            jsd = 0

        assert 0 <= jsd <= 1, f"JSD is {jsd}"

        return jsd

    def compute_out_of_road_rate(self):
        """
        Compute the rate of agents that are out of the road. I.e., those who has OFF_ROAD as death cause.

        :return fraction of agents that are out of the road.
        """

        out_of_road_agents = [(aid, agent) for (aid, agent) in self.__agents.items() if
                              agent["death_cause"] == DeathCause.OFF_ROAD.value]
        off_road_rate = len(out_of_road_agents) / len(self.__agents)

        return off_road_rate

    def compute_collision_rate(self):
        """
        Compute the rate of agents that have collided. I.e., those who has COLLISION as death cause.
        For each agent that collided, find the time in which it collided in the simulation vs in the dataset so that
        we can compare it
        :return: fraction of agents that have collided.
        """

        collision_agents = [(aid, agent) for (aid, agent) in self.__agents.items() if
                            agent["death_cause"] == DeathCause.COLLISION.value]
        collision_rate = len(collision_agents) / len(self.__agents)

        return collision_rate

    def compute_jsd_vel(self):
        """Compute the two velocity distributions using the JSD"""
        real_speeds, simulated_speeds = self.get_simulated_real_speeds()

        all_simulated_speeds = []
        all_real_speeds = []

        for agent_id, features in self.__agents.items():
            all_simulated_speeds.extend(simulated_speeds[agent_id])
            all_real_speeds.extend(real_speeds[agent_id])

        jsd_vel = self.compute_jsd(all_simulated_speeds, all_real_speeds)

        return jsd_vel

    def compute_interference(self):
        """
        Compute the average number of interferences for all agents.
        """

        interferences = 0
        for agent_id, features in self.__agents.items():
            interferences += len(features["interference"])

        return np.mean(interferences)

    def compute_coverage_maps(self, gt_agents_data: Dict[str, Any], filename: str):
        """
        Plot the maps and all the trajectories of the agents.
        """

        def plot_trajectories(ax, x, y, color, alpha=0.1):
            sns.scatterplot(x=x, y=y, ax=ax, alpha=alpha, color=color)

        def setup_plot(ax, title, xlabel="x", ylabel="y"):
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

        episode_name = list(self.__agents.keys())[0].split("/")[0]
        map_file = f"scenarios/data/automatum/{episode_name}/staticWorld.xodr"
        map = Map.parse_from_opendrive(map_file)
        plt.rcParams["axes.labelsize"] = 20

        fig, axs = plt.subplots(1, 2, figsize=(20, 10), sharey=True)
        plot_map(map, markings=True, ax=axs[0])
        plot_map(map, markings=True, ax=axs[1])

        for agent_id, features in self.__agents.items():
            agent_episode_name = agent_id.split("/")[0]
            if agent_episode_name != episode_name:
                continue

            real_x = gt_agents_data[agent_id].x_vec
            real_y = gt_agents_data[agent_id].y_vec
            simulated_x = [state.position.x for state in features["states"]]
            simulated_y = [state.position.y for state in features["states"]]

            plot_trajectories(axs[0], real_x, real_y, "green")
            plot_trajectories(axs[1], simulated_x, simulated_y, "orange")

        sns.set(font_scale=1.5)
        setup_plot(axs[0], "Real Trajectories")
        setup_plot(axs[1], "Simulated Trajectories")

        fig.suptitle("Coverage Density Estimation of Real and Simulated Trajectories", fontsize=20)
        plt.savefig(f"evaluation/plots/coverage_density_{filename}.svg", format="svg")

        def plot_single_map(title, filename_suffix, plot_real=True, plot_simulated=True):
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            plot_map(map, markings=True, ax=ax)

            for agent_id, features in self.__agents.items():
                agent_episode_name = agent_id.split("/")[0]
                if agent_episode_name != episode_name:
                    continue

                if plot_real:
                    real_x = gt_agents_data[agent_id].x_vec
                    real_y = gt_agents_data[agent_id].y_vec
                    plot_trajectories(ax, real_x, real_y, "green")

                if plot_simulated:
                    simulated_x = [state.position.x for state in features["states"]]
                    simulated_y = [state.position.y for state in features["states"]]
                    plot_trajectories(ax, simulated_x, simulated_y, "orange")

            setup_plot(ax, title)
            plt.savefig(f"evaluation/plots/{filename_suffix}_{filename}.svg", format="svg")

        plot_single_map("Simulated Trajectories", "simulated_coverage_density", plot_real=False)
        plot_single_map("Real Trajectories", "real_coverage_density", plot_simulated=False)
        plot_single_map("Real and Simulated Trajectories", "real_simulated_coverage_density", plot_real=True,
                        plot_simulated=True)

    def compute_similarity_scores(self, gt_agents_data: Dict[str, Any], filename: str = "default"):

        if len(self.__agents) == 0:
            return {}, {}

        # Safety
        collision_rate = self.compute_collision_rate()
        off_road_rate = self.compute_out_of_road_rate()

        # Diversity
        self.compute_coverage_maps(gt_agents_data, filename)

        # Realism
        jsd_vel = self.compute_jsd_vel()
        interferences = self.compute_interference()

        # Final Scores
        metric_values = {
            "Collision_rate": collision_rate,
            "Off_road_rate": off_road_rate,
            "JSD_Velocity": jsd_vel,
            "Interference": interferences
        }

        return metric_values

    @staticmethod
    def plot_distogram(label, real_data=None, simulated_data=None):
        # Plot histograms and PDFs for real and simulated data
        plt.figure(figsize=(6, 4))

        # Plot real data
        if real_data is not None:
            plt.hist(real_data, bins=30, density=True, alpha=0.6, color='blue', label=f'Real {label}')

        # Plot simulated data
        if simulated_data is not None:
            plt.hist(simulated_data, bins=30, density=True, alpha=0.6, color='orange', label=f'Simulated {label}')

        plt.xlabel(label)
        plt.ylabel('Normalized PDF')
        plt.legend()
        plt.grid(True, which='both', axis='both', color='black', linestyle='--', linewidth=0.5)

    @property
    def agents(self):
        return self.__agents

    def load(self, agents):
        self.__agents = agents

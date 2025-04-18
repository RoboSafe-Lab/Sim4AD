import json
import logging
from collections import defaultdict
from typing import Dict, Any, List
import numpy as np
from scipy.spatial import distance
from stable_baselines3 import SAC

from sim4ad.data import ScenarioConfig, DatasetScenario
from sim4ad.opendrive import plot_map, Map
from sim4ad.path_utils import get_agent_id_combined, get_config_path
from simulator.policy_agent import PolicyAgent
from simulator.simulator_util import DeathCause
from simulator.state_action import State
from sim4ad.offlinerlenv.td3bc_automatum import Actor
from simulator.simulator_util import PositionNearbyAgent as PNA
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def get_clusters(scenario_name):
    with open(f"scenarios/configs/{scenario_name}_drivingStyle.json", "rb") as f:
        return json.load(f)


class EvaluationFeaturesExtractor:

    def __init__(self, sim_name: str, episode_names: List[str] = None):
        self.__sim_name = sim_name

        # A dictionary (indexed by agent id) of dictionaries (indexed by feature type) of features over time.
        self.__agents = defaultdict(lambda: defaultdict(list))
        self.__computed_metrics = defaultdict(dict)  # Store them as [metric][agent_id] = value
        if episode_names is not None:
            gt_agents_data = self.load_ground_truth_data(episode_names)
            self.__real_closest_distances, self.__real_speeds, self.__real_ttcs, self.__real_tths = self.get_ground_truth_info(
                gt_agents_data)

    @staticmethod
    def load_ground_truth_data(episode_names):
        """ Given a list of episodes, get the value for each agent by combining the episode_id and the agent_id """
        gt_agents_data = {}

        for episode_name in episode_names:
            scenario_name = episode_name.split("-")[2]
            config = ScenarioConfig.load(get_config_path(scenario_name))
            data_loader = DatasetScenario(config)
            episode = data_loader.load_episode(episode_id=episode_name)

            for agent_id, agent in episode.agents.items():
                gt_agents_data[get_agent_id_combined(episode_name=episode_name, agent_id=agent_id)] = agent

        return gt_agents_data

    @staticmethod
    def get_ground_truth_info(gt_agents_data):
        """Get the real closest, closest distances, speeds, ttcs and tths"""
        # fidelity metrics
        real_closest_distances = {}
        real_speeds = {}
        real_ttcs = {}
        real_tths = {}

        for agent_id, agent in gt_agents_data.items():
            real_speeds[agent_id] = {"vx_s": agent.vx_vec, "vy_s": agent.vy_vec}
            real_ttcs[agent_id] = agent.ttc_dict_vec
            real_tths[agent_id] = agent.tth_dict_vec
            closest_dis = []
            episode_id = agent_id.split("/")[0]
            for inx, t in enumerate(agent.time):
                try:
                    surrounding_agents = agent.object_relation_dict_list[inx]
                except IndexError:
                    break
                dis = np.inf
                for surrounding_agent_relation, surrounding_agent_id in surrounding_agents.items():
                    if surrounding_agent_id is not None:
                        surrounding_agent = gt_agents_data[get_agent_id_combined(episode_id, surrounding_agent_id)]
                        long_distance, lat_distance = agent.get_lat_and_long(t, surrounding_agent)
                        if dis > np.sqrt(long_distance**2 + lat_distance**2):
                            dis = np.sqrt(long_distance**2 + lat_distance**2)

                if not np.isinf(dis):
                    closest_dis.append(dis)
            real_closest_distances[agent_id] = closest_dis

        return real_closest_distances, real_speeds, real_ttcs, real_tths

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

        self.__agents[aid]["death_cause"] = death_cause.value
        self.__agents[aid]["states"] = states
        self.__agents[aid]["observations"] = obs
        # self.agents[agent.agent_id]["actions"] = actions

        # compute ttcs and tths
        for i in range(len(states)):
            self.compute_ttc_tth(agent, states[i], agent.nearby_vehicles[i], episode_id, add=True)

        self.__agents[aid]["nearby_vehicles"] = agent.nearby_vehicles
        self.__agents[aid]["midline_distance"] = agent.distance_midline
        self.__agents[aid]["time"] = [state.time for state in states]
        if isinstance(agent.policy, str):
            self.__agents[aid]["policy_type"] = agent.policy
        elif isinstance(agent.policy, SAC):
            self.__agents[aid]["policy_type"] = "sac"
        elif isinstance(agent.policy, Actor):
            self.__agents[aid]["policy_type"] = "offlinerl"
        else:
            self.__agents[aid]["policy_type"] = agent.policy.name

        self.__agents[aid]["interference"] = agent.interferences

    def get_picklable_agents(self):
        """
        Return the agents in a format that can be pickled.
        """
        return dict(self.__agents)

    # def cluster_agents(self, agents, cluster):
    #     """Return agents of a specific cluster"""
    #     if cluster is None:
    #         return agents
    #
    #     cluster_per_episode = {}
    #     agents_in_cluster = {}
    #     for agent_id in agents:
    #         scenario_name = agent_id.split("-")[2]
    #         if scenario_name not in cluster_per_episode:
    #             cluster_per_episode[scenario_name] = get_clusters(scenario_name)
    #         if cluster_per_episode[scenario_name][agent_id] == cluster:
    #             agents_in_cluster[agent_id] = agents[agent_id]
    #     return agents_in_cluster

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
        ttc = {PNA.CENTER_IN_FRONT: None, PNA.CENTER_BEHIND: None}
        tth = {PNA.CENTER_IN_FRONT: None, PNA.CENTER_BEHIND: None}
        closest_d = np.inf

        for position, nearby_agent in nearby_vehicles.items():
            # TTC and TTH are computed for the agents in front and rear
            if position in [PNA.CENTER_IN_FRONT, PNA.CENTER_BEHIND] and nearby_agent is not None:
                d = (state.position.distance(nearby_agent["position"]) - agent.meta.length / 2 -
                     nearby_agent["metadata"].length / 2)
                v_ego = abs(state.velocity)
                v_other = abs(nearby_agent["velocity"])

                if position == PNA.CENTER_IN_FRONT:
                    ttc[position] = d / (v_ego - v_other)
                    if v_ego:
                        tth[position] = d / v_ego
                else:
                    ttc[position] = d / (v_other - v_ego)
                    if v_other:
                        tth[position] = d / v_other
                # keep the same value as the dataset
                if ttc[position] < 0:
                    ttc[position] = -1
                # agents overlapped, collision occurs
                if tth[position] < 0:
                    tth[position] = 0
            # save the closest distance to nearby vehicles
            d = (state.position.distance(nearby_agent["position"])) if nearby_agent is not None else None
            if d is not None and closest_d > d:
                closest_d = d

        if add:
            aid = get_agent_id_combined(episode_id, agent.agent_id)
            self.__agents[aid]["TTC"].append(ttc)
            self.__agents[aid]["TTH"].append(tth)
            self.__agents[aid]["closest_dis"].append(closest_d)
        else:
            return ttc, tth

    def get_simulated_real_speeds(self):
        """Get the real and simulated speed for distribution analysis"""
        real_speeds = []
        simulated_speeds = []

        for agent_id, features in self.__agents.items():
            real_vel_x = self.__real_speeds[agent_id]["vx_s"]
            real_vel_y = self.__real_speeds[agent_id]["vy_s"]
            real_speed = np.sqrt(np.array(real_vel_x) ** 2 + np.array(real_vel_y) ** 2)
            simulated_speed = np.array([state.velocity for state in features["states"]])

            real_speeds.extend(real_speed)
            simulated_speeds.extend(simulated_speed)

        return real_speeds, simulated_speeds

    def compute_all_ittc_tth(self, cluster=None):
        """
        Compute the inverse TTC (iTTC) and TTH distributions of the agents.
        """
        all_simulated_ittcs = []
        all_simulated_tths = []
        all_real_ittcs = []
        all_real_tths = []


        # agents = self.cluster_agents(self.__agents, cluster)
        for agent_id, features in self.__agents.items():
            for v in features["TTC"]:
                if v[PNA.CENTER_IN_FRONT] is not None:
                    ittc = 1.0 / v[PNA.CENTER_IN_FRONT]
                    # > 1 means critical
                    if ittc > 1:
                        ittc = 1.0
                    # < 0 means uncritical
                    elif ittc < 0:
                        ittc = 0
                    all_simulated_ittcs.append(ittc)
            all_simulated_tths.extend(v[PNA.CENTER_IN_FRONT] for v in features["TTH"] if v[PNA.CENTER_IN_FRONT] is not None)

            # keep the same as real speed, obtain data from self.__agents.
            for v in self.__real_ttcs[agent_id]:
                if v['front_ego'] is not None:
                    ittc = 1.0 / v['front_ego']
                    if ittc > 1:
                        ittc = 1.0
                    elif ittc < 0:
                        ittc = 0
                    all_real_ittcs.append(ittc)

            all_real_tths.extend(v['front_ego'] for v in self.__real_tths[agent_id] if v['front_ego'] is not None)

        return all_simulated_ittcs, all_real_ittcs, all_simulated_tths, all_real_tths

    def get_simulated_real_closest_dis(self):
        """Get the real and simulated closest distance for distribution analysis"""
        real_closest_dis = []
        simulated_closest_dis = []

        for agent_id, features in self.__agents.items():
            real_dis = self.__real_closest_distances[agent_id]
            simulated_closest_dis.extend([d for d in features["closest_dis"] if not np.isinf(d)])
            real_closest_dis.extend(real_dis)

        return real_closest_dis, simulated_closest_dis

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
    @staticmethod
    def compute_hellinger_distance(simulated_values, real_values):
        """
        Compute the Hellinger distance between two distributions given their simulated and real values.
        """
        # Compute the histograms of the distributions
        nr_bins = 10
        simulated_hist, _ = np.histogram(simulated_values, bins=nr_bins, density=True)
        real_hist, _ = np.histogram(real_values, bins=nr_bins, density=True)

        # Normalize histograms
        simulated_hist = simulated_hist + 1e-6
        real_hist = real_hist + 1e-6

        simulated_hist = simulated_hist / np.sum(simulated_hist)
        real_hist = real_hist / np.sum(real_hist)

        # Compute the Hellinger distance
        hellinger_distance = (1 / np.sqrt(2)) * np.sqrt(
            np.sum((np.sqrt(simulated_hist) - np.sqrt(real_hist)) ** 2)
        )

        assert 0 <= hellinger_distance <= 1, f"Hellinger Distance is {hellinger_distance}"

        return hellinger_distance


    def compute_out_of_road_rate(self):
        """
        Compute the rate of agents that are out of the road. I.e., those who has OFF_ROAD as death cause.

        :return fraction of agents that are out of the road.
        """

        out_of_road_agents = [(aid, agent) for (aid, agent) in self.__agents.items() if
                              agent["death_cause"] == DeathCause.OFF_ROAD.value]
        off_road_rate = len(out_of_road_agents) / len(self.__agents)

        return off_road_rate
    
    def compute_reach_goal_rate(self):

        reach_goal_agents = [(aid, agent) for (aid, agent) in self.__agents.items() if
                              agent["death_cause"] == DeathCause.GOAL_REACHED.value]
        reach_goal_rate = len(reach_goal_agents) / len(self.__agents)
        return reach_goal_rate

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
        all_real_speeds, all_simulated_speeds = self.get_simulated_real_speeds()

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

    def plot_speed_distribution(self):
        """Plot speed distributions for human likeness analysis"""
        real_speeds, simulated_speeds = self.get_simulated_real_speeds()
        self.plot_distogram(label='Speed (m/s)', real_data=real_speeds, simulated_data=simulated_speeds)

        plt.show()

    def plot_criticality_distribution(self, cluster=None):
        """plot criticality distribution for diversity analysis"""
        all_simulated_ittcs, all_real_ittcs, all_simulated_tths, all_real_tths = self.compute_all_ittc_tth(cluster=cluster)

        self.plot_distogram_2(label='iTTC (1/s)', real_data=all_real_ittcs, simulated_data=all_simulated_ittcs)
        self.plot_distogram(label='THW (s)', real_data=all_real_tths, simulated_data=all_simulated_tths)

        plt.show()

    def plot_closest_dis_distribution(self):
        """Plot the closest distance distribution for human likeness analysis"""
        real_closest_dis, simulated_closest_dis = self.get_simulated_real_closest_dis()
        self.plot_distogram(label='Distance (m)', real_data=real_closest_dis, simulated_data=simulated_closest_dis)

        plt.show()
    
    def plot_distogram_2(self, label, real_data=None, simulated_data=None):
                # Plot histograms and PDFs for real and simulated data
        plt.figure(figsize=(6, 4))
        
        # Plot real data
        if real_data is not None:
            plt.hist(real_data, bins=30, density=True, alpha=0.6, color='#2171B5', label=f'Real {label}')#qian blue #2171B5

        # Plot simulated data
        if simulated_data is not None:
            plt.hist(simulated_data, bins=30, density=True, alpha=0.6, color='#FF6F61', label=f'Simulated {label}')#qian cheng #E67E22 #qianlv #66BB6A #qian hong #FF6F61 

        if simulated_data is not None and real_data is not None:
            jsd = self.compute_jsd(simulated_data, real_data)
            hellinger = self.compute_hellinger_distance(simulated_data, real_data)
            #plt.text(0.58, 0.77, f'JSD: {jsd:.4f}\nHellinger Distance: {hellinger:.4f}', transform=plt.gca().transAxes,
                    # verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
            plt.text(0.57, 0.77, f'JSD: {jsd:.4f}\nHellinger Distance: {hellinger:.4f}', 
                    transform=plt.gca().transAxes,
                    verticalalignment='top', 
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', linewidth=0.8),
                    linespacing=1.5)
        plt.xlabel(label)
        plt.xlim(0,0.5)
        plt.ylabel('Normalized PDF')
        plt.legend()
        plt.grid(True, which='both', axis='both', color='black', linestyle='--', linewidth=0.5)

    def plot_distogram(self, label, real_data=None, simulated_data=None):
        # Plot histograms and PDFs for real and simulated data
        plt.figure(figsize=(7, 5))
        
        # Plot real data
        if real_data is not None:
            plt.hist(real_data, bins=30, density=True, alpha=0.6, color='#2171B5', label=f'Real {label}')#blue 2171B5
            
        # Plot simulated data
        if simulated_data is not None:
            plt.hist(simulated_data, bins=30, density=True, alpha=0.6, color='#FF6F61', label=f'Simulated {label}')#orange E67E22 #green 66BB6A #pink FF6F61

        if simulated_data is not None and real_data is not None:
            jsd = self.compute_jsd(simulated_data, real_data)
            hellinger = self.compute_hellinger_distance(simulated_data, real_data)
            #plt.text(0.58, 0.77, f'JSD: {jsd:.4f}\nHellinger Distance: {hellinger:.4f}', transform=plt.gca().transAxes,
                    # verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
            plt.text(0.75, 0.77, f'JSD: {jsd:.4f}\nHD: {hellinger:.4f}', 
                    transform=plt.gca().transAxes,
                    verticalalignment='top', 
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', linewidth=0.8),
                    linespacing=1.5)
        plt.xlabel(label)
        plt.ylabel('Normalized PDF')
        plt.legend()
        plt.grid(True, which='both', axis='both', color='black', linestyle='--', linewidth=0.5)

    @property
    def agents(self):
        return self.__agents

    def load(self, agents):
        self.__agents = agents

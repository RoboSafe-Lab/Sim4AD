import logging
import pickle
from collections import defaultdict
from typing import Dict, Any

import numpy as np
from scipy.spatial import distance
from shapely import LineString
from stable_baselines3 import SAC

from sim4ad.data import Episode
from sim4ad.opendrive import plot_map, Map
from sim4ad.path_utils import get_agent_id_combined
from sim4ad.util import Box
from simulator.policy_agent import PolicyAgent
from simulator.simulator_util import DeathCause
from simulator.state_action import State
from simulator.simulator_util import PositionNearbyAgent as PNA
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class EvaluationFeaturesExtractor:

    def __init__(self, sim_name: str):
        self.__sim_name = sim_name

        # A dictionary (indexed by agent id) of dictionaries (indexed by feature type) of features over time.
        self.__agents = defaultdict(lambda: defaultdict(list))
        self.__computed_metrics = defaultdict(dict)  # Store them as [metric][agent_id] = value

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
        self.agents[aid]["right_marking_distance"] = agent.distance_right_lane_marking
        self.agents[aid]["left_marking_distance"] = agent.distance_left_lane_marking
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
                v_ego = abs(state.speed)
                v_other = abs(nearby_agent["speed"])

                if v_other < v_ego:
                    ttc = d / (v_ego - v_other)
                else:
                    # A collision is impossible if the other agent is faster than the ego
                    ttc = -1

                tth = d / v_ego

            ttcs[position] = ttc
            tths[position] = tth

        if add:
            aid = get_agent_id_combined(episode_id, agent.agent_id)
            self.__agents[aid]["TTC"].append(ttcs)
            self.__agents[aid]["TTH"].append(tths)
        else:
            return ttcs, tths

    def rmse_speed(self, real_speedsxy: Dict[str, Dict[str, np.ndarray]]):
        """
        Compute the (average) root-mean-square error (RMSE) of the speed of ALL agents, compared to the dataset.

        :param real_speeds: The real speeds of the agents in the dataset. Dict[agent_id, speeds] where speeds is a
        dictionary of speeds, indexed as (vx_s, vy_s), and the values are numpy arrays of the speeds.
        """

        rmse = 0

        real_speeds, simulated_speeds = self.get_simulated_real_speeds(real_speedsxy)

        for agent_id, features in self.__agents.items():
            real_speed = real_speeds[agent_id]
            simulated_speed = simulated_speeds[agent_id]
            rmse += np.sqrt(np.mean((real_speed - simulated_speed) ** 2))

        return rmse / len(self.__agents)

    def get_simulated_real_speeds(self, real_speedsxy: Dict[str, Dict[str, np.ndarray]]):

        real_speeds = {}
        simulated_speeds = {}

        for agent_id, features in self.__agents.items():
            real_vel_x = real_speedsxy[agent_id]["vx_s"]
            real_vel_y = real_speedsxy[agent_id]["vy_s"]
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

    def rmse_position(self, real_position: Dict[str, Dict[str, np.ndarray]]):
        """
        Compute the (average) root-mean-square error (RMSE) of the position of ALL agents, compared to the dataset.
        """

        rmse = 0

        real_xs, real_ys, simulated_xs, simulated_ys = self.get_real_simulated_xy(real_position)
        for agent_id, features in self.__agents.items():
            real_x = real_xs[agent_id]
            real_y = real_ys[agent_id]
            simulated_x = simulated_xs[agent_id]
            simulated_y = simulated_ys[agent_id]

            rmse_x = np.sqrt(np.mean((real_x - simulated_x) ** 2))
            rmse_y = np.sqrt(np.mean((real_y - simulated_y) ** 2))

            rmse += rmse_y + rmse_x

        return rmse / len(self.__agents)

    def compute_ade_all_agents(self, real_position: Dict[str, Dict[str, np.ndarray]]):
        """
        Compute the standard Average Displacement Error (ADE) for all agents.
        """

        real_xs, real_ys, simulated_xs, simulated_ys = self.get_real_simulated_xy(real_position)
        ades = []
        for agent_id, features in self.__agents.items():
            real_x = real_xs[agent_id]
            real_y = real_ys[agent_id]
            simulated_x = simulated_xs[agent_id]
            simulated_y = simulated_ys[agent_id]

            # count the number of nan values in the simulated positions (i.e., when the agent is dead)
            # nr_nan = np.sum(np.isnan(simulated_x))
            # if nr_nan > 0:
            #     # Compare the real and simulated positions only up to the point where the agent is dead.
            #     real_x = real_x[:-nr_nan]
            #     real_y = real_y[:-nr_nan]
            #     simulated_x = simulated_x[:-nr_nan]
            #     simulated_y = simulated_y[:-nr_nan]
            #
            # ADE is the L2 norm of the difference between the real and simulated positions.
            ade = np.linalg.norm(np.array([real_x, real_y]) - np.array([simulated_x, simulated_y]), axis=0)
            assert len(ade) == len(real_x), f"Length of ADE: {len(ade)}, length of real_x: {len(real_x)}"
            #
            # # ensure there are no nan values in the ADE
            assert not np.isnan(ade).any(), f"Agent {agent_id} has nan values in ADE"
            # if nr_nan > 0:
            #     # take the last value of the ADE and set it to gamma.
            #     gamma = 0.9
            #     # create a list with nr_nans gammas raised to the power of the number of nan values, ranging from 1 to the number of nan values
            #     gamma_list = [gamma ** i * ade[-1] for i in range(nr_nan)]
            #     ade = np.concatenate([ade, gamma_list])

            if len(ade) == 0:
                continue
            else:
                ades.append(np.mean(ade))

        self.__computed_metrics["ADE"] = np.mean(ades)
        return np.mean(ades), ades

    def compute_fde_all_agents(self, real_position: Dict[str, Dict[str, np.ndarray]]):
        """
        Compute the Final Displacement Error (FDE) for all agents.
        """

        real_xs, real_ys, simulated_xs, simulated_ys = self.get_real_simulated_xy(real_position)
        fdes = []
        for agent_id, features in self.__agents.items():
            real_x = real_xs[agent_id]
            real_y = real_ys[agent_id]
            simulated_x = simulated_xs[agent_id]
            simulated_y = simulated_ys[agent_id]

            if len(real_x) == 0 or len(simulated_x) == 0:
                continue

            # FDE is the L2 norm of the difference between the final real and simulated positions.
            fde = np.linalg.norm(np.array([real_x[-1], real_y[-1]]) - np.array([simulated_x[-1], simulated_y[-1]]))

            fdes.append(fde)

        self.__computed_metrics["FDE"] = np.mean(fdes)
        return np.mean(fdes)

    def compute_td_ade_all_agents(self, real_position: Dict[str, Dict[str, np.ndarray]]):
        """
        Compute the Temporal-Difference Average Displacement Error (TD-ADE) for all agents.
        """
        # For all agents, take their (x, y) positions and compute the 1-step displacement for each step, i.e.,
        # the direction vector in which they moved in one step. Then, compare this to the real direction vector.

        real_xs, real_ys, simulated_xs, simulated_ys = self.get_real_simulated_xy(real_position)
        td_ades = []
        for aid, agent in self.__agents.items():
            real_x = real_xs[aid]
            real_y = real_ys[aid]
            simulated_x = simulated_xs[aid]
            simulated_y = simulated_ys[aid]

            plot = False
            if plot:
                import matplotlib.pyplot as plt
                plt.plot(real_x, real_y, label="Real")
                plt.plot(simulated_x, simulated_y, label="Simulated")
                plt.legend()
                plt.show()

            td_ade_i = []
            for i in range(5, len(real_x)):
                real_direction = np.array([real_x[i] - real_x[i - 5], real_y[i] - real_y[i - 5]])
                simulated_direction = np.array(
                    [simulated_x[i] - simulated_x[i - 5], simulated_y[i] - simulated_y[i - 5]])

                td_ade_i.append(np.linalg.norm(real_direction - simulated_direction)** 2)

            if len(td_ade_i) == 0:
                continue
            else:
                td_ades.append(np.mean(td_ade_i))

        self.__computed_metrics["TD-ADE"] = np.mean(td_ades)
        return np.mean(td_ades), td_ades

    @staticmethod
    def _get_sim_d_nearby_one_vehicle(features):
        distances = []
        for t in range(len(features["states"])):
            distances_at_t = []
            bbox_agent = LineString(features["states"][t].bbox.boundary)
            for position, nearby_agent in features["nearby_vehicles"][t].items():
                if nearby_agent is not None:
                    # compute the distances from the bbox of the agent i at t and its nearby vehicles
                    bbox_neighbour = LineString(Box(nearby_agent["position"], nearby_agent["metadata"].length,
                                                    nearby_agent["metadata"].width, nearby_agent["heading"]).boundary)
                    distances_at_t.append(bbox_agent.distance(bbox_neighbour))
            if len(distances_at_t) > 0:
                distances.append(np.mean(distances_at_t))

        if len(distances) == 0:
            return []
        return distances

    @staticmethod
    def _get_real_d_nearby_one_vehicle(gt_data_all_agents, aid):
        gt = gt_data_all_agents[aid]
        distances = []
        for t in range(len(gt.time)):
            distances_at_t = []
            bbox_agent = LineString(Box(np.array([gt.x_vec[t], gt.y_vec[t]]), gt.length, gt.width, gt.psi_vec[t]).boundary)
            try:
                for position, nearby_agent in gt.object_relation_dict_list[t].items():
                    if nearby_agent is not None:
                        episode_name = aid.split("/")[0]
                        nearby_agent_data = gt_data_all_agents[get_agent_id_combined(episode_name, nearby_agent)]
                        t_nearby = nearby_agent_data.time.index(gt.time[t])
                        bbox_neighbour = LineString(Box(np.array([nearby_agent_data.x_vec[t_nearby], nearby_agent_data.y_vec[t_nearby]]),
                                                        nearby_agent_data.length,
                                                        nearby_agent_data.width,nearby_agent_data.psi_vec[t_nearby]).boundary)
                        distances_at_t.append(bbox_agent.distance(bbox_neighbour))
            except IndexError:
                pass
            if len(distances_at_t) > 0:
                distances.append(np.mean(distances_at_t))

        if len(distances) == 0:
            return []
        return distances

    def distance_with_nearby_vehicles(self, gt_agents_data: Dict[str, Any]):
        """
        Compute the 10th percentiel fo the average distance to the nearby vehicles.
        """

        # 1. compute the average distance to the nearby vehicles
        distances_simulated = []
        distances_real = []
        for agent_id, features in self.__agents.items():
            distances_simulated.extend(self._get_sim_d_nearby_one_vehicle(features))
            distances_real.extend(self._get_real_d_nearby_one_vehicle(gt_agents_data, agent_id))

        # 2. compute the 10th percentile
        sim_distances_10p = np.percentile(distances_simulated, 10)
        real_distances_10p = np.percentile(distances_real, 10)

        score = 1 - np.abs(np.tanh(np.abs(sim_distances_10p - real_distances_10p) / real_distances_10p))

        return sim_distances_10p, real_distances_10p, score

    def compute_ttc_tth_jsd(self, real_ttcs: Dict[str, Dict[str, np.ndarray]],
                            real_tths: Dict[str, Dict[str, np.ndarray]]):
        """
        Compute the Jensen-Shannon Divergence between the TTC and TTH distributions of the agents.
        """

        all_simulated_ttcs = []
        all_simulated_tths = []
        all_real_ttcs = []
        all_real_tths = []

        # Take the average ttc and tth for each agent at each time step
        for agent_id, features in self.__agents.items():

            T_ttc = min(len(features["TTC"]), len(real_ttcs[agent_id]))
            T_tth = min(len(features["TTH"]), len(real_tths[agent_id]))

            for t in range(T_ttc):
                ttcs_t = features["TTC"][t]
                real_ttcs_t = real_ttcs[agent_id][t]

                # Compute the average ttc and tth for the agent at time t
                avg_ttc_t = np.mean([ttc for ttc in ttcs_t.values() if (ttc is not None)])
                avg_real_ttc_t = np.mean([ttc for ttc in real_ttcs_t.values() if (ttc is not None)])

                if not np.isnan(avg_ttc_t):
                    all_simulated_ttcs.append(avg_ttc_t)
                if not np.isnan(avg_real_ttc_t):
                    all_real_ttcs.append(avg_real_ttc_t)

            for t in range(T_tth):
                tths_t = features["TTH"][t]
                real_tths_t = real_tths[agent_id][t]

                # it is -1 when no vehicle is in front
                avg_real_tth_t = np.mean([tth for tth in real_tths_t.values() if (tth is not None)])
                avg_tth_t = np.mean([tth for tth in tths_t.values() if (tth is not None)])

                if not np.isnan(avg_tth_t):
                    all_simulated_tths.append(avg_tth_t)
                if not np.isnan(avg_real_tth_t):
                    all_real_tths.append(avg_real_tth_t)

        jsd_ttc = self.compute_jsd(all_simulated_ttcs, all_real_ttcs)
        jsd_tth = self.compute_jsd(all_simulated_tths, all_real_tths)

        def score_tt(jsd):
            return 1 - jsd

        return jsd_ttc, jsd_tth, score_tt(jsd_ttc), score_tt(jsd_tth)

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
    def _compute_cr_or_score(list_agents, gt_agents):
        """
        :param list_agents: should be either out_of_road_agents or collision_agents
        """
        scores_list = []
        for agent_id, agent in list_agents:
            original_agent = gt_agents[agent_id]
            t_gt = original_agent.time[-1]
            try:
                t_sim = agent["time"][-1]
            except IndexError:
                if len(agent["time"]) == 0:
                    # agent immediately died
                    t_sim = 0

            alpha = np.min([np.abs(1 - t_sim / t_gt), 1])
            assert 0 <= alpha <= 1, f"Alpha is {alpha}"

            beta = 0.001
            scores_list.append(beta ** alpha)

        if len(scores_list) == 0:
            return 1

        score = np.mean(scores_list)
        assert 0 <= score <= 1, f"Score is {score}"
        return score

    def compute_out_of_road_rate(self, gt_agents_data: Dict[str, Any]):
        """
        Compute the rate of agents that are out of the road. I.e., those who has OFF_ROAD as death cause.

        :return fraction of agents that are out of the road.
        """

        out_of_road_agents = [(aid, agent) for (aid, agent) in self.__agents.items() if
                              agent["death_cause"] == DeathCause.OFF_ROAD.value]
        off_road_rate = len(out_of_road_agents) / len(self.__agents)
        off_road_score = self._compute_cr_or_score(out_of_road_agents, gt_agents_data)

        return off_road_rate, off_road_score

    def compute_collision_rate(self, gt_agents_data: Dict[str, Any]):
        """
        Compute the rate of agents that have collided. I.e., those who has COLLISION as death cause.
        For each agent that collided, find the time in which it collided in the simulation vs in the dataset so that
        we can compare it
        :return: fraction of agents that have collided.
        """

        collision_agents = [(aid, agent) for (aid, agent) in self.__agents.items() if
                            agent["death_cause"] == DeathCause.COLLISION.value]
        collision_rate = len(collision_agents) / len(self.__agents)

        collision_score = self._compute_cr_or_score(collision_agents, gt_agents_data)
        return collision_rate, collision_score

    def compute_distance_right_marking_metric(self, gt_data_agents):
        """
        Average the standard deviation of the distances to the right lane markings of a single agent across
        all agents across times.
        """

        sim_distances = []
        real_distances = []
        for aid, agent in self.__agents.items():
            if len(agent["right_marking_distance"]) == 0:
                continue
            sim_distances.append(np.std(agent["right_marking_distance"]))
            real_distances.append(np.std(gt_data_agents[aid].distance_right_lane_marking))

        mean_sim_distance = np.mean(sim_distances)
        mean_real_distance = np.mean(real_distances)

        score = 1 - np.abs(np.tanh(np.abs(mean_sim_distance - mean_real_distance) / mean_real_distance))

        return mean_sim_distance, mean_real_distance, score

    def compute_jsd_vel(self, real_speeds: Dict[str, Dict[str, np.ndarray]]):

        real_speeds, simulated_speeds = self.get_simulated_real_speeds(real_speeds)

        all_simulated_speeds = []
        all_real_speeds = []

        for agent_id, features in self.__agents.items():
            all_simulated_speeds.extend(simulated_speeds[agent_id])
            all_real_speeds.extend(real_speeds[agent_id])

        jsd_vel = self.compute_jsd(all_simulated_speeds, all_real_speeds)

        def score_vel(jsd):
            return 1 - jsd

        return jsd_vel, score_vel(jsd_vel)

    def compute_interference(self):
        """
        Compute the average number of interferences for all agents.
        """

        interferences = 0
        for agent_id, features in self.__agents.items():
            interferences += len(features["interference"])

        score = 1 - np.tanh(interferences / len(self.__agents))
        return np.mean(interferences), score

    def compute_coverage_maps(self, gt_agents_data: Dict[str, Any], filename: str):
        """
        Plot the maps and all the trajectories of the agents.
        """

        # Compare side by side the real and simulated trajectories plotting them on the map of the environment
        # for each agent
        episode_name = list(self.__agents.keys())[0].split("/")[0]
        map_file = f"scenarios/data/automatum/{episode_name}/staticWorld.xodr"
        map = Map.parse_from_opendrive(map_file)
        plt.rcParams["axes.labelsize"] = 20

        # create two subplots one for the real trajectories and one for the simulated trajectories
        fig, axs = plt.subplots(1, 2, figsize=(20, 10), sharey=True)

        # for each plot set the map
        plot_map(map, markings=True, ax=axs[0])
        plot_map(map, markings=True, ax=axs[1])

        for agent_id, features in self.__agents.items():

            agent_episode_name = agent_id.split("/")[0]
            if agent_episode_name != episode_name:
                continue

            # Do a scatter plot of the real and simulated trajectories using a very light color
            real_x = gt_agents_data[agent_id].x_vec
            real_y = gt_agents_data[agent_id].y_vec
            simulated_x = [state.position.x for state in features["states"]]
            simulated_y = [state.position.y for state in features["states"]]

            sns.scatterplot(x=real_x, y=real_y, ax=axs[0], alpha=0.1, color="green")
            sns.scatterplot(x=simulated_x, y=simulated_y, ax=axs[1], alpha=0.1, color="orange")

        # Set a big font
        sns.set(font_scale=1.5)
        axs[0].set_title("Real Trajectories")
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("y")
        axs[1].set_title("Simulated Trajectories")
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("y")

        fig.suptitle("Coverage Density Estimation of Real and Simulated Trajectories", fontsize=20)
        plt.savefig(f"evaluation/plots/coverage_density_{filename}.svg", format="svg")

        ### Repeat the same but with one plot for real
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        plot_map(map, markings=True, ax=ax)
        for agent_id, features in self.__agents.items():
            agent_episode_name = agent_id.split("/")[0]
            if agent_episode_name != episode_name:
                continue

            simulated_x = [state.position.x for state in features["states"]]
            simulated_y = [state.position.y for state in features["states"]]

            sns.scatterplot(x=simulated_x, y=simulated_y, ax=ax, alpha=0.1, color="orange")

        ax.set_title("Simulated Trajectories")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.savefig(f"evaluation/plots/simulated_coverage_density_{filename}.svg", format="svg")

        ## DO the same for reals
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        plot_map(map, markings=True, ax=ax)
        for agent_id, features in self.__agents.items():
            agent_episode_name = agent_id.split("/")[0]
            if agent_episode_name != episode_name:
                continue

            real_x = gt_agents_data[agent_id].x_vec
            real_y = gt_agents_data[agent_id].y_vec

            sns.scatterplot(x=real_x, y=real_y, ax=ax, alpha=0.1, color="green")

        ax.set_title("Real Trajectories")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.savefig(f"evaluation/plots/real_coverage_density_{filename}.svg", format="svg")

        # One combined
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        plot_map(map, markings=True, ax=ax)
        for agent_id, features in self.__agents.items():
            agent_episode_name = agent_id.split("/")[0]
            if agent_episode_name != episode_name:
                continue

            real_x = gt_agents_data[agent_id].x_vec
            real_y = gt_agents_data[agent_id].y_vec
            simulated_x = [state.position.x for state in features["states"]]
            simulated_y = [state.position.y for state in features["states"]]

            sns.scatterplot(x=real_x, y=real_y, ax=ax, alpha=0.1, color="green")
            sns.scatterplot(x=simulated_x, y=simulated_y, ax=ax, alpha=0.1, color="orange")

        ax.set_title("Real and Simulated Trajectories")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.savefig(f"evaluation/plots/real_simulated_coverage_density_{filename}.svg", format="svg")


    def compare_td_ade(self, ades, td_ades, gt_agents, filename):
        # take the vehicel in which the difference is the highest and then plot the trajecotry of the real and simulated
        max_diff = 0
        max_diff_agent = None
        for i, agent_id in enumerate(self.__agents.keys()):
            if i == len(td_ades) - 1:
                break
            diff = np.abs(ades[i] - td_ades[i])
            if diff > max_diff:
                max_diff = diff
                max_diff_agent = agent_id

        if max_diff_agent is None:
            return

        agent_episode_name = max_diff_agent.split("/")[0]
        map_file = f"scenarios/data/automatum/{agent_episode_name}/staticWorld.xodr"
        map = Map.parse_from_opendrive(map_file)
        plt.rcParams["axes.labelsize"] = 20

        # plot real trajecotery in blue and simulated in orange using seaborn on the same mao
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        plot_map(map, markings=True, ax=ax)

        real_x = gt_agents[max_diff_agent].x_vec
        real_y = gt_agents[max_diff_agent].y_vec
        simulated_x = [state.position.x for state in self.__agents[max_diff_agent]["states"]]
        simulated_y = [state.position.y for state in self.__agents[max_diff_agent]["states"]]

        sns.scatterplot(x=real_x, y=real_y, ax=ax, alpha=1, color="blue", linewidth=0)
        sns.scatterplot(x=simulated_x, y=simulated_y, ax=ax, alpha=1, color="orange", linewidth=0)
        filename = filename + max_diff_agent.split("/")[1] + "_" + str(max_diff)
        plt.savefig(f"evaluation/plots/td_ade_comparison_{filename}.svg", format="svg")


    def compute_similarity_scores(self, gt_agents_data: Dict[str, Any], filename: str = "default"):

        if len(self.__agents) == 0:
            return {}, {}

        ### 1. Compute the Metrics ####

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

        rmse_speed = self.rmse_speed(real_speedsxy=real_speeds)
        rmse_position = self.rmse_position(real_position=real_position)

        # Fidelity
        ade, ades = self.compute_ade_all_agents(real_position=real_position)
        s_ade = self.score_fidelity(ade)
        fde = self.compute_fde_all_agents(real_position=real_position)
        s_fde = self.score_fidelity(fde)
        td_ade, td_ades = self.compute_td_ade_all_agents(real_position=real_position)
        s_td_ade = self.score_fidelity(td_ade)

        self.compare_td_ade(ades=ades, td_ades=td_ades, gt_agents=gt_agents_data, filename=filename)

        # Safety
        collision_rate, s_collision = self.compute_collision_rate(gt_agents_data)
        off_road_rate, s_off_road = self.compute_out_of_road_rate(gt_agents_data)
        sim_distances_10p, real_distances_10p, s_distance_nearby = self.distance_with_nearby_vehicles(gt_agents_data)

        # Diversity
        mean_sim_distance, mean_real_distance, s_right_marking = self.compute_distance_right_marking_metric(gt_agents_data)
        self.compute_coverage_maps(gt_agents_data, filename)

        # Realism
        jsd_ttc, jsd_tth, s_ttc, s_tth = self.compute_ttc_tth_jsd(real_ttcs=real_ttcs, real_tths=real_tths)
        jsd_vel, s_velocity = self.compute_jsd_vel(real_speeds=real_speeds)
        interferences, s_interference = self.compute_interference()

        assert 0 <= s_ade <= 1, f"Score ADE is {s_ade}"
        assert 0 <= s_fde <= 1, f"Score FDE is {s_fde}"
        assert 0 <= s_td_ade <= 1, f"Score TD-ADE is {s_td_ade}"
        assert 0 <= s_collision <= 1, f"Score Collision is {s_collision}"
        assert 0 <= s_off_road <= 1, f"Score Off-road is {s_off_road}"
        assert 0 <= s_distance_nearby <= 1, f"Score Distance nearby is {s_distance_nearby}"
        assert 0 <= s_ttc <= 1, f"Score TTC is {s_ttc}"
        assert 0 <= s_tth <= 1, f"Score TTH is {s_tth}"
        assert 0 <= s_velocity <= 1, f"Score Velocity is {s_velocity}"
        assert 0 <= s_interference <= 1, f"Score Interference is {s_interference}"
        assert 0 <= s_right_marking <= 1, f"Score Right marking is {s_right_marking}"

        ### Final Scores ###
        metric_values = {
            #"RMSE_speed": rmse_speed,
            #"RMSE_position": rmse_position,
            "ADE": ade,
            "FDE": fde,
            "TD-ADE": td_ade,
            "Collision_rate": collision_rate,
            "Off_road_rate": off_road_rate,
            "Distance_nearby": sim_distances_10p,
            "Distance_right_marking": mean_sim_distance,

            "JSD_TTC": jsd_ttc,
            "JSD_TTH": jsd_tth,
            "JSD_Velocity": jsd_vel,
            "Interference": interferences
        }

        scores = {
            "S_ADE": s_ade,
            "S_FDE": s_fde,
            "S_TD_ADE": s_td_ade,
            "S_TTC": s_ttc,
            "S_TTH": s_tth,
            "S_Collision": s_collision,
            "S_Off_road": s_off_road,
            "S_Distance_nearby": s_distance_nearby,
            "S_Distance_right_marking": s_right_marking,
            "S_Velocity": s_velocity,
            "S_Interference": s_interference
        }

        return metric_values, scores

    @staticmethod
    def score_fidelity(x):
        # ade, fde, td-ade
        return 1 - np.tanh(x)

    @property
    def agents(self):
        return self.__agents

    def load(self, agents):
        self.__agents = agents

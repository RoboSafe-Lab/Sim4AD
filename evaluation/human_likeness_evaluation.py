import gymnasium as gym
import numpy as np
from tqdm import tqdm
import torch
import os
import pickle
from loguru import logger

from sim4ad.path_utils import get_config_path, get_file_name_trajectories, get_path_offlinerl_model, get_path_sac_model
from sim4ad.data import ScenarioConfig
from simulator.lightweight_simulator import Sim4ADSimulation
from sim4ad.offlinerlenv.td3bc_automatum import TD3_BC_Loader, EvalConfig
from baselines.sac.model import Actor as SACActor
from evaluation.evaluation_functions import EvaluationFeaturesExtractor


class RLEvaluation:
    """Evaluate offline RL and SAC for human likeness analysis"""

    def __init__(self, actor, episode_names, spawn_method, policy_type, clustering):
        driving_styles = {
            "Aggressive": actor,
            "Normal": actor,
            "Cautious": actor
        }

        self.actor = actor
        self.env = gym.make("SimulatorEnv-v0", dataset_split="test")
        self.sim = Sim4ADSimulation(episode_name=episode_names, spawn_method=spawn_method, policy_type=policy_type,
                                    clustering=clustering, driving_style_policies=driving_styles)

    def simulation(self, visualization: bool = False):
        """Using the simulator to simulate trajectories of all agents"""
        self.sim.full_reset()
        # done = False # TODO: uncomment this to run until we use all vehicles
        # while not done:
        #     assert spawn_method != "random", "we will never finish!"
        #     done = sim.step(return_done=True)

        simulation_length = 50  # seconds
        for _ in tqdm(range(int(np.floor(simulation_length / self.sim.dt)))):
            self.sim.step()

        # remove all agents left in the simulation.
        self.sim.kill_all_agents()
        if visualization:
            self.sim.replay_simulation()

        simulation_agents = self.sim.evaluator.get_picklable_agents()

        return simulation_agents


def begin_evaluation(simulation_agents, evaluation_episodes):
    """Evaluate the human likeness using speed, criticality distributions"""
    evaluator = EvaluationFeaturesExtractor("evaluator", evaluation_episodes)
    evaluator.load(simulation_agents)
    evaluator.plot_criticality_distribution()
    evaluator.plot_speed_distribution()
    # evaluator.plot_distance_distribution()


def main():
    # configuration
    policy_type = 'offlinerl'  # "bc-all-obs-5_pi_cluster_Aggressive"  # "bc-all-obs-1.5_pi" "idm"
    normal_map = "appershofen"
    spawn_method = "dataset_all"
    clustering = "all"

    configs = ScenarioConfig.load(get_config_path(normal_map))
    idx = configs.dataset_split["test"]
    evaluation_episodes = [x.recording_id for i, x in enumerate(configs.episodes) if i in idx]

    output_dir = get_file_name_trajectories(policy_type, spawn_method, irl_weights=None,
                                            episode_name=evaluation_episodes, param_config=None)

    # Check if the results are already saved
    if os.path.exists(output_dir):
        with open(output_dir, "rb") as f:
            simulation_agents = pickle.load(f)
        # Begin the evaluation function
        logger.info('Diversity evaluation started!')
        begin_evaluation(simulation_agents, evaluation_episodes)
    else:
        simulation_agents = None
        parameter_loader = TD3_BC_Loader(EvalConfig)

        if policy_type == 'offlinerl':
            model_path = get_path_offlinerl_model()
            parameter_loader.load_model(model_path)
            actor = parameter_loader.actor
            offline_rl_eva = RLEvaluation(actor=actor, episode_names=evaluation_episodes, spawn_method=spawn_method,
                                          policy_type=policy_type, clustering=clustering)
            simulation_agents = offline_rl_eva.simulation()

        # TODO: add other policies
        elif policy_type == 'sac':
            device = parameter_loader.config.device
            actor = SACActor(parameter_loader.env, device=device).to(device)
            actor.load_state_dict(torch.load(get_path_sac_model()))
            sac_eval = RLEvaluation(actor=actor, episode_names=evaluation_episodes, spawn_method=spawn_method,
                                    policy_type=policy_type, clustering=clustering)
            simulation_agents = sac_eval.simulation()

        elif policy_type == 'bc':
            raise NotImplementedError

        # Create the directory if it does not exist
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        with open(output_dir, "wb") as f:
            pickle.dump(simulation_agents, f)

        logger.info('Trajectories saved!')


if __name__ == "__main__":
    main()

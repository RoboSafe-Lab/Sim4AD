from loguru import logger
import pickle
import torch
import os
from typing import Dict

from sim4ad.data import ScenarioConfig
from sim4ad.path_utils import get_config_path, get_path_offlinerl_model, get_file_name_trajectories, get_path_sac_model
from sim4ad.offlinerlenv.td3bc_automatum import TD3_BC_Loader, EvalConfig, wrap_env
from evaluation.evaluation_functions import EvaluationFeaturesExtractor
from baselines.sac.model import Actor as SACActor


class RLEvaluation:
    """Evaluate the offline rl for its diversity"""

    def __init__(self, actor, trainer_loader: TD3_BC_Loader):
        """

        :param actor: The actor of the policy we want to evaluate
        :param trainer_loader: the trainer loader which contains the environment and the normalization parameters
        """
        self.actor = actor
        self.env = wrap_env(trainer_loader.env, state_mean=trainer_loader.state_mean,
                            state_std=trainer_loader.state_std,
                            reward_mean=trainer_loader.reward_mean, reward_std=trainer_loader.reward_std,
                            reward_normalization=True)

    @torch.no_grad()
    def simulation(self, spawn_method: str, visualization: bool = False) -> Dict:
        """Using the policy to generate trajectories"""
        self.actor.eval()
        looped_dataset = False
        steps = 0

        while not looped_dataset:
            obs, info = self.env.reset(seed=EvalConfig.seed)
            terminated, truncated = False, False
            while not terminated and not truncated:
                # TODO: @Cheng, you had action = self.actor.act(obs, TrainConfig.device)
                action = self.actor.act(obs, device=EvalConfig.device, deterministic=True)
                state, reward, terminated, truncated, _ = self.env.step(action)
                obs = state

                steps += 1
                if steps % 1000 == 0:
                    logger.info(f"Simulation time {self.env.unwrapped.simulation.time} in mode "
                                f"{spawn_method} w/ offline rl policy")

            looped_dataset = self.env.unwrapped.simulation.done_full_cycle
            # show the rollout of the policy
            if visualization:
                self.env.unwrapped.simulation.replay_simulation()

        self.env.unwrapped.simulation.kill_all_agents()

        simulation_agents = self.env.unwrapped.simulation.evaluator.get_picklable_agents()

        return simulation_agents


def begin_evaluation(simulation_agents: Dict):
    """Evaluate the criticality distribution under the current policy for diversity analysis"""
    evaluator = EvaluationFeaturesExtractor("evaluator")
    evaluator.load(simulation_agents)
    evaluator.plot_criticality_distribution()


def main():
    # configuration
    policy_type = 'sac'
    normal_map = "appershofen"
    spawn_method = "dataset_all"

    configs = ScenarioConfig.load(get_config_path(normal_map))
    idx = configs.dataset_split["test"]
    evaluation_episodes = [x.recording_id for i, x in enumerate(configs.episodes) if i in idx]

    output_dir = get_file_name_trajectories(policy_type, spawn_method, None, episode_name=evaluation_episodes, param_config=EvalConfig)
    # Check if the results are already saved
    if os.path.exists(output_dir):
        with open(output_dir, "rb") as f:
            simulation_agents = pickle.load(f)
        # Begin the evaluation function
        begin_evaluation(simulation_agents)
    else:
        simulation_agents = None

        # The loader is needed to get the normalization parameters of the environment
        model_path = get_path_offlinerl_model()
        parameter_loader = TD3_BC_Loader(EvalConfig)

        if policy_type == 'offlinerl':
            parameter_loader.load_model(model_path)
            actor = parameter_loader.actor
            offline_rl_eva = RLEvaluation(actor, parameter_loader)
            simulation_agents = offline_rl_eva.simulation(spawn_method=spawn_method)

        # TODO: add other policies
        elif policy_type == 'sac':
            device = parameter_loader.config.device
            actor = SACActor(parameter_loader.env, device=device).to(device)
            actor.load_state_dict(torch.load(get_path_sac_model()))
            sac_eval = RLEvaluation(actor, parameter_loader)
            simulation_agents = sac_eval.simulation(spawn_method=spawn_method)

        elif policy_type == 'bc':
            raise NotImplementedError

        # Create the directory if it does not exist
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        with open(output_dir, "wb") as f:
            pickle.dump(simulation_agents, f)


if __name__ == "__main__":
    main()

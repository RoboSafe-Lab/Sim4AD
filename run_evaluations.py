import pickle
from dataclasses import dataclass, field
from typing import Dict
from loguru import logger

import gymnasium as gym
from enum import Enum
import os
import torch

from evaluation.diversity_evaluation import DiversityEvaluation
from evaluation.human_likeness_evaluation import HumanLikenessEvaluation
from sim4ad.data import ScenarioConfig
from sim4ad.path_utils import get_config_path, get_path_offlinerl_model, get_file_name_trajectories, get_path_sac_model
from sim4ad.offlinerlenv.td3bc_automatum import TD3_BC_Loader, wrap_env, get_normalisation_parameters
from evaluation.evaluation_functions import EvaluationFeaturesExtractor
from baselines.sac.model import Actor as SACActor


### TODO: CHANGE THIS ACCORDING TO THE EVALUATION YOU WANT TO RUN ###
class PolicyType(Enum):
    # !!! Make sure that the values of the enums below are all different across policy types!!!
    # Otherwise, the enum will not initialize all the values correctly
    SAC_BASIC_REWARD = {"Aggressive": "best_model_sac_Aggressive_irlFalse_SimulatorEnv-v0__model__1__1723173376.pth",
                        "Normal": "",
                        "Cautious": "best_model_sac_Cautious_irlFalse_SimulatorEnv-v0__model__1__1723191164.pth",
                        "all": ""}
    SAC_IRL_REWARD = {"Aggressive": "best_model_sac_Aggressive_irlTrue_SimulatorEnv-v0__model__1__1723133135.pth",
                      "Normal": "best_model_sac_Normal_irlTrue_SimulatorEnv-v0__model__1__1723133135.pth",
                      "Cautious": "best_model_sac_Cautious_irlTrue_SimulatorEnv-v0__model__1__1723133135.pth",
                      "all": "best_model_sac_all_irlTrue_SimulatorEnv-v0__model__1__1723192188.pth"}
    OFFLINERL = "offlinerl"
    BC = "bc"


class EvaluationType(Enum):
    # !!! Make sure that the values of the enums below are all different across evaluation types!!!
    DIVERSITY = {"spawn_method": "dataset_one", "clusters": ["Normal", "Cautious", "all"]} # TODO: add Aggressive
    HUMAN_LIKENESS = {"spawn_method": "dataset_all", "clusters": ["all"]}
    GENERALIZATION = "generalization"  # TODO


@dataclass
class EvalConfig:
    policies_to_evaluate: list = (PolicyType.SAC_IRL_REWARD, PolicyType.SAC_BASIC_REWARD)
    evaluation_to_run = EvaluationType.HUMAN_LIKENESS

    env_name: str = "SimulatorEnv-v0"
    test_map: str = "appershofen"
    generalization_map: str = "brunn"
    seed: int = 15648  # todo: change this?

    state_normalization: bool = True
    reward_normalization: bool = True  # only used if state_normalization is True

    ### DO NOT CHANGE THE FOLLOWING PARAMETERS -- They are set automatically based on the evaluation performed ###
    dataset_split: str = "test"  # Set depending on the type of evaluation done
    spawn_method: str = None  # Set depending on the type of evaluation done
    use_irl_reward: bool = None  # Set depending on the type of evaluation done
    device: str = field(default_factory=lambda: torch.device("cuda") if torch.cuda.is_available() else "cpu")
    clusters: list = None  # Set in EvaluationType

    def __str__(self):
        return str(self.__dict__)


### SHOULD *NOT* NEED TO CHANGE ###

# Modify the parameters of EvalConfig as needed
POLICY_CONFIGS = {
    PolicyType.SAC_BASIC_REWARD: {"reward_normalization": False, "use_irl_reward": False},
    PolicyType.SAC_IRL_REWARD: {"reward_normalization": True, "use_irl_reward": True},
}


def load_policy(policy_type: PolicyType, cluster: str, env: gym.Env, device):
    if policy_type == PolicyType.OFFLINERL:
        raise NotImplementedError("adaptation needed!")  # TODO: @ Cheng
        model_path = get_path_offlinerl_model()
        parameter_loader = TD3_BC_Loader(eval_config)
        parameter_loader.load_model(model_path)
        return parameter_loader.actor

    elif policy_type == PolicyType.SAC_BASIC_REWARD or policy_type == PolicyType.SAC_IRL_REWARD:
        actor = SACActor(env, device=device).to(device)
        actor.load_state_dict(torch.load(policy_type.value[cluster]))
        return actor

    elif policy_type == PolicyType.BC:
        raise NotImplementedError
    else:
        raise NotImplementedError(f"Policy type {policy_type} not implemented")


def begin_evaluation(simulation_agents, evaluation_episodes):
    """Evaluate the human likeness using speed, criticality distributions"""
    evaluator = EvaluationFeaturesExtractor("evaluator", evaluation_episodes)
    evaluator.load(simulation_agents)
    evaluator.plot_criticality_distribution()
    evaluator.plot_speed_distribution()
    evaluator.plot_closest_dis_distribution()


def main():

    VISUALIZATION = False  # Set to True if you want to visualize the simulation while saving the trajectories
    for policy in EvalConfig.policies_to_evaluate:
        # Concatenate the configs for the evaluation type and the policy
        eval_configs = EvalConfig(**POLICY_CONFIGS[policy], **EvalConfig.evaluation_to_run.value)
        logger.info(f"Evaluation with params: {eval_configs}")

        for map_to_use in [eval_configs.test_map, eval_configs.generalization_map]:
            map_configs = ScenarioConfig.load(get_config_path(map_to_use))
            idx = map_configs.dataset_split[eval_configs.dataset_split]
            evaluation_episodes = [x.recording_id for i, x in enumerate(map_configs.episodes) if i in idx]

            for cluster in eval_configs.clusters:

                output_dir = get_file_name_trajectories(experiment_name=eval_configs.evaluation_to_run.name,
                                                        map_name=map_to_use, policy_type=policy.name, cluster=cluster,
                                                        irl_weights=eval_configs.use_irl_reward,
                                                        spawn_method=eval_configs.spawn_method,
                                                        episode_names=evaluation_episodes,
                                                        dataset_split=eval_configs.dataset_split,
                                                        state_normalization=eval_configs.state_normalization,
                                                        reward_normalization=eval_configs.reward_normalization)

                # Check if the results are already saved
                if os.path.exists(output_dir):
                    with open(output_dir, "rb") as f:
                        simulation_agents = pickle.load(f)
                    # Begin the evaluation function
                    logger.info(f'{eval_configs.evaluation_to_run.name} evaluation started!')
                    begin_evaluation(simulation_agents, evaluation_episodes)
                else:
                    if eval_configs.evaluation_to_run == EvaluationType.DIVERSITY:
                        evaluator = DiversityEvaluation(eval_configs, evaluation_episodes, policy, cluster, load_policy)
                    elif eval_configs.evaluation_to_run == EvaluationType.HUMAN_LIKENESS:
                        assert cluster == "all", "Human likeness evaluation only supports 'all' cluster"
                        evaluator = HumanLikenessEvaluation(eval_configs, evaluation_episodes, policy, cluster,
                                                            driving_style_model_paths=policy.value,
                                                            load_policy=load_policy)
                    else:
                        raise NotImplementedError

                    simulation_agents = evaluator.simulation(visualization=VISUALIZATION)

                    # Create the directory if it does not exist
                    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
                    with open(output_dir, "wb") as f:
                        pickle.dump(simulation_agents, f)

                    logger.info(f'Trajectories for {cluster} in map {map_to_use} saved!')


if __name__ == "__main__":
    main()

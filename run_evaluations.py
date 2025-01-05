import pickle
from dataclasses import dataclass, field
from typing import List
from loguru import logger

import gymnasium as gym
from enum import Enum
import os
import torch

from evaluation.trajectory_extractor import TrajectoryExtractor
from sim4ad.data import ScenarioConfig
from sim4ad.path_utils import get_config_path, get_path_offlinerl_model, get_file_name_trajectories, get_path_sac_model
from sim4ad.offlinerlenv.td3bc_automatum import TD3_BC_Loader, wrap_env, get_normalisation_parameters, TrainConfig
from evaluation.evaluation_functions import EvaluationFeaturesExtractor
from baselines.sac.model import Actor as SACActor
from baselines.bc_baseline import BCBaseline as BC


class PolicyType(Enum):
    """ Used to define the different types of policies that can be evaluated. Each evaluation
     type should have a different value for each of the enums below. """
    SAC_BASIC_REWARD = "sac_basic_reward"
    SAC_IRL_REWARD = "sac_irl_reward"
    OFFLINERL = "offlinerl"
    BC = "bc"


"""
### CHANGE THIS ACCORDING TO THE EVALUATION YOU WANT TO RUN ###
Add for each cluster the path to the model that vehicles in that cluster should follow
"""
EVAL_POLICIES = {
    PolicyType.SAC_BASIC_REWARD: {
        #"Aggressive": "best_model_sac_Aggressive_irlFalse_SimulatorEnv-v0__model__1__1724203527.pth",
        #"Aggressive": "best_model_sac_Aggressive_irlFalse_SimulatorEnv-v0__model__1__1735322548.pth",
        "Aggressive": "best_model_sac_Aggressive_irlFalse_SimulatorEnv-v0__model__1__1735480317.pth",
        #"Normal": "best_model_sac_Normal_irlFalse_SimulatorEnv-v0__model__1__1734188839.pth",
        "Normal": "best_model_sac_multi_MultiAgentSAC__1__1734876090.pth",
       # "Normal": "best_model_sac_Normal_irlFalse_SimulatorEnv-v0__model__1__1735479265.pth",
        #"Cautious": "best_model_sac_Cautious_irlFalse_SimulatorEnv-v0__model__1__1724207833.pth",
        "Cautious": "best_model_sac_Cautious_irlFalse_SimulatorEnv-v0__model__1__1735479606.pth"},
        #"All": "best_model_sac_Cautious_irlFalse_SimulatorEnv-v0__model__1__1724207833.pth"},
    PolicyType.SAC_IRL_REWARD: {
        "Aggressive": "best_model_sac_Aggressive_irlTrue_SimulatorEnv-v0__model__1__1724161974.pth",
        "Normal": "best_model_sac_Normal_irlTrue_SimulatorEnv-v0__model__1__1724161974.pth",
        "Cautious": "best_model_sac_Cautious_irlTrue_SimulatorEnv-v0__model__1__1724161974.pth",
        "All": "best_model_sac_All_irlTrue_SimulatorEnv-v0__model__1__1724161974.pth"},
    PolicyType.OFFLINERL: {
        "Aggressive": "results/offlineRL/Aggressive_checkpoint.pt",
        "Normal": "results/offlineRL/Normal_checkpoint.pt",
        "Cautious": "results/offlineRL/Cautious_checkpoint.pt"},
        #"All": "results/offlineRL/All_checkpoint.pt"},
    PolicyType.BC: {
        "Aggressive": "bc-all-obs-5_pi_cluster_Aggressive",
        "Normal": "bc-all-obs-5_pi_cluster_Normal",
        "Cautious": "bc-all-obs-5_pi_cluster_Cautious",
        "All": "bc-all-obs-5_pi_cluster_All"}
}


class EvaluationType(Enum):
    # !!! Make sure that the values of the enums below are all different across evaluation types!!!
    DIVERSITY = {"spawn_method": "dataset_all", "clusters": ["Aggressive"]}
    HUMAN_LIKENESS = {"spawn_method": "dataset_all", "clusters": ["Aggressive"]}
    GENERALIZATION = "generalization"  # TODO


@dataclass
class EvalConfig:
    """ PARAMETERS FOR THE EVALUATION """
    policies_to_evaluate: str = "SAC_BASIC_REWARD"  # e.g., "sac_basic_reward-sac_irl_reward-offlinerl-bc"
    evaluation_to_run = f"{EvaluationType.DIVERSITY.name}-{EvaluationType.HUMAN_LIKENESS.name}"

    env_name: str = "SimulatorEnv-v0"
    test_map: str = "appershofen"
    generalization_map: str = "appershofen"
    seed: int = 15648  # todo: change this?

    state_normalization: bool = True
    reward_normalization: bool = True  # only used if state_normalization is True

    ### DO NOT CHANGE THE FOLLOWING PARAMETERS -- They are set automatically based on the evaluation performed ###
    dataset_split: str = "test"  # Set depending on the type of evaluation done
    spawn_method: str = None  # Set depending on the type of evaluation done
    use_irl_reward: bool = None  # Set depending on the type of evaluation done
    device: str = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    clusters: list = None  # Set in EvaluationType

    def __str__(self):
        return str(self.__dict__)


### SHOULD *NOT* NEED TO CHANGE ###

# Modify the parameters of EvalConfig as needed
POLICY_CONFIGS = {
    PolicyType.SAC_BASIC_REWARD: {"reward_normalization": False, "use_irl_reward": False},
    PolicyType.SAC_IRL_REWARD: {"reward_normalization": True, "use_irl_reward": True},
    PolicyType.OFFLINERL: {"reward_normalization": True, "use_irl_reward": True},
    PolicyType.BC: {"reward_normalization": False, "use_irl_reward": False}
}


def load_policy(policy_type: PolicyType, env: gym.Env, device, eval_configs: EvalConfig,
                evaluation_episodes: List, model_path: str = None):
    """IF YOU ADD A POLICY TYPE, MAKE SURE TO ADD IT TO THE ENUMS ABOVE TOO"""
    if policy_type == PolicyType.OFFLINERL:
        parameter_loader = TD3_BC_Loader(config=TrainConfig, env=env, dummy=True)
        parameter_loader.load_model(model_path)
        actor = parameter_loader.actor
        actor.eval()
        return actor

    elif policy_type == PolicyType.SAC_BASIC_REWARD or policy_type == PolicyType.SAC_IRL_REWARD:
        actor = SACActor(env, device=device).to(device)
        actor.load_state_dict(torch.load(model_path))
        actor.eval()
        return actor

    elif policy_type == PolicyType.BC:
        return BC(name=model_path, evaluation=True)
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
    # TODO: if visualisation is true, `simulation_length` should be low (in trajectory_extractor.py) to avoid long
    #  waiting time
    VISUALIZATION = True  # Set to True if you want to visualize the simulation while saving the trajectories
    policies_to_evaluate = EvalConfig.policies_to_evaluate.split("-")
    evaluation_methods = EvalConfig.evaluation_to_run.split("-")

    for evaluation in evaluation_methods:
        evaluation = EvaluationType[evaluation.upper()]
        for policy in policies_to_evaluate:
            policy = PolicyType[policy.upper()]
            # Concatenate the configs for the evaluation type and the policy
            eval_configs = EvalConfig(**POLICY_CONFIGS[policy],
                                      **evaluation.value)
            logger.info(f"Evaluation with params: {eval_configs}")

            for map_to_use in [eval_configs.test_map, eval_configs.generalization_map]:
                map_configs = ScenarioConfig.load(get_config_path(map_to_use))
                idx = map_configs.dataset_split[eval_configs.dataset_split]
                evaluation_episodes = [x.recording_id for i, x in enumerate(map_configs.episodes) if i in idx]

                for cluster in eval_configs.clusters:
                    output_dir = get_file_name_trajectories(experiment_name=evaluation.name,
                                                            map_name=map_to_use, policy_type=policy.name,
                                                            cluster=cluster,
                                                            irl_weights=eval_configs.use_irl_reward,
                                                            spawn_method=eval_configs.spawn_method,
                                                            episode_names=evaluation_episodes,
                                                            dataset_split=eval_configs.dataset_split,
                                                            state_normalization=eval_configs.state_normalization,
                                                            reward_normalization=eval_configs.reward_normalization)

                    # Check if the trajectories are already saved
                    if os.path.exists(output_dir):
                        with open(output_dir, "rb") as f:
                            simulation_agents = pickle.load(f)
                        # Begin the evaluation function
                        logger.info(f'{evaluation.name} evaluation started for {cluster} policy!')
                        begin_evaluation(simulation_agents, evaluation_episodes)
                    else:
                        if evaluation == EvaluationType.DIVERSITY:
                            # All the vehicles in the simulation, regardless sof the ground truth cluster, will have the
                            # same policy
                            driving_style_model_paths = EVAL_POLICIES[policy]
                            driving_style_model_paths = {c: driving_style_model_paths[cluster]
                                                         for c in driving_style_model_paths}
                        elif evaluation == EvaluationType.HUMAN_LIKENESS:
                            assert cluster == "All", "Human likeness evaluation only supports 'All' cluster. Got {" \
                                                     "cluster}"
                            driving_style_model_paths = EVAL_POLICIES[policy]
                        else:
                            raise NotImplementedError(f"Evaluation type {evaluation} not implemented")

                        # The cluster below is "All" rather than `cluster` because we use `cluster` to load the
                        # correct policy, but then `All` the vehicles will have that policy
                        evaluator = TrajectoryExtractor(eval_configs, evaluation_episodes, policy_type=policy,
                                                        cluster="All", load_policy=load_policy,
                                                        driving_style_model_paths=driving_style_model_paths)
                        simulation_agents = evaluator.simulation(visualization=VISUALIZATION)

                        # Create the directory if it does not exist
                        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
                        with open(output_dir, "wb") as f:
                            pickle.dump(simulation_agents, f)

                        logger.info(f'Trajectories for {cluster} in map {map_to_use} saved!')


if __name__ == "__main__":
    main()


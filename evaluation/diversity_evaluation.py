from loguru import logger
import torch
from typing import Dict
import gymnasium as gym

from sim4ad.offlinerlenv.td3bc_automatum import get_normalisation_parameters, wrap_env


class DiversityEvaluation:
    """Evaluate the offline rl and SAC for its diversity"""

    def __init__(self, eval_configs, evaluation_episodes, policy, cluster, load_policy):
        """

        :param actor: The actor of the policy we want to evaluate
        :param trainer_loader: the trainer loader which contains the environment and the normalization parameters
        """
        env = gym.make(eval_configs.env_name, episode_names=evaluation_episodes, seed=eval_configs.seed,
                       clustering=cluster, dataset_split=eval_configs.dataset_split,
                       use_irl_reward=eval_configs.use_irl_reward, spawn_method=eval_configs.spawn_method)

        if eval_configs.state_normalization:
            means = get_normalisation_parameters(driving_style=env.unwrapped.simulation.clustering,
                                                 map_name=env.unwrapped.map_name,
                                                 dataset_split=env.unwrapped.dataset_split,
                                                 state_dim=env.observation_space.shape[0])

            state_mean, state_std, reward_mean, reward_std = means
            env = wrap_env(env=env, state_mean=state_mean, state_std=state_std, reward_mean=reward_mean,
                           reward_std=reward_std, reward_normalization=eval_configs.reward_normalization)
        self.actor = load_policy(policy, cluster, env, device=eval_configs.device,
                                 eval_configs=eval_configs, evaluation_episodes=evaluation_episodes)

        self.eval_configs = eval_configs
        self.env = env

    @torch.no_grad()
    def simulation(self, visualization: bool = False) -> Dict:
        """Using the policy to generate trajectories"""
        self.actor.eval()
        looped_dataset = False
        steps = 0

        while not looped_dataset:
            obs, info = self.env.reset(seed=self.eval_configs.seed)
            terminated, truncated = False, False
            while not terminated and not truncated:
                action = self.actor.act(obs, device=self.eval_configs.device, deterministic=True)
                state, reward, terminated, truncated, _ = self.env.step(action)
                obs = state

                steps += 1
                if steps % 1000 == 0:
                    logger.info(f"Simulation time {self.env.unwrapped.simulation.time} in mode "
                                f"{self.eval_configs.spawn_method} in diversity evaluation")

            looped_dataset = self.env.unwrapped.simulation.done_full_cycle
            # show the rollout of the policy
            if visualization:
                self.env.unwrapped.simulation.replay_simulation(save=False)

        self.env.unwrapped.simulation.kill_all_agents()

        simulation_agents = self.env.unwrapped.simulation.evaluator.get_picklable_agents()

        self.env.close()

        return simulation_agents


if __name__ == "__main__":
    raise NotImplementedError("This script is not meant to be run directly. Please run `run_evaluations.py` instead.")

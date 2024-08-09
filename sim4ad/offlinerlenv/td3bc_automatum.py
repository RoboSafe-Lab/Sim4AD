import copy
import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from loguru import logger
from tqdm import tqdm

import gymnasium as gym
import gym_env
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import pickle
from sim4ad.common_constants import MISSING_NEARBY_AGENT_VALUE
from sim4ad.data import ScenarioConfig
from sim4ad.path_utils import get_config_path

TensorBatch = List[torch.Tensor]


@dataclass
class TrainConfig:
    # REQUIRED
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_irl_reward: bool = True  # Use IRL reward; if False, use basic reward (1 for reach goal, -1 for collisions)
    env: str = "SimulatorEnv-v0"  # OpenAI gym environment name
    dataset_split: str = "train"  # Dataset split to use
    normalize: bool = True  # get mean and std of state AND reward
    normalize_reward: bool = True  # Normalize reward
    spawn_method: str = None # whether to use the gym env
    # need to be changed according to the policy to be trained
    map_name: str = "appershofen"
    driving_style: str = "Aggressive"

    # TD3 + BC training-specific parameters
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_freq: int = 10  # How often (time steps) we evaluate
    n_episodes: int = 10  # How many episodes run during evaluation
    max_timesteps: int = 5000  # Max time steps to run environment
    checkpoints_path: Optional[str] = 'results/offlineRL'  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load
    # TD3
    buffer_size: int = 2_000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount
    expl_noise: float = 0.1  # Std of Gaussian exploration noise
    tau: float = 0.005  # Target network update rate
    policy_noise: float = 0.2  # Noise added to target actor during critic update
    noise_clip: float = 0.5  # Range to clip target actor noise
    policy_freq: int = 2  # Frequency of delayed actor updates
    # TD3 + BC
    alpha: float = 2.5  # Coefficient for Q function in actor loss

    # Wandb logging
    project: str = "CORL"
    group: str = "TD3_BC-Automatum"
    name: str = "TD3_BC"

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    # Create a mask for valid values (those that are not MISSING_NEARBY_AGENT_VALUE)
    mask = states != MISSING_NEARBY_AGENT_VALUE

    # Normalize the states element-wise where mask is True
    for i in range(states.shape[1]):  # Iterate over each column
        col_mask = mask[:, i]  # Mask for the current column
        states[col_mask, i] = (states[col_mask, i] - mean[i]) / std[i]
    return states


def normalized_rewards(rewards: np.ndarray, mean: float, std: float):
    for i in range(rewards.shape[0]):
        rewards[i] = (rewards[i] - mean) / std
        rewards[i] = np.clip(rewards[i], -1, 1)
    return rewards


def wrap_env(
        env: gym.Env,
        state_mean: Union[np.ndarray, float] = 0.0,
        state_std: Union[np.ndarray, float] = 1.0,
        reward_mean: float = 0.0,
        reward_std: float = 1.0,
        reward_normalization: bool = False,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        if state is None:
            return state
        else:
            state = np.array(state).reshape(1, -1)
            # Mask the states what do not have a nearby agent
            mask = state != MISSING_NEARBY_AGENT_VALUE
            for i in range(state.shape[1]):
                col_mask = mask[:, i]
                state[col_mask, i] = (state[col_mask, i] - state_mean[i]) / state_std[i]
            return state

    def normalize_reward(reward):
        reward = (reward - reward_mean) / reward_std  # epsilon should be already added in std.
        reward = np.clip(reward, -1, 1)
        return float(reward)

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_normalization:
        env = gym.wrappers.TransformReward(env, normalize_reward)
    return env


class ReplayBuffer:
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            buffer_size: int,
            device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in tensor format, i.e. from Dict[str, np.array].
    def load_automatum_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def add_transition(self):
        # Use this method to add new data into the replay buffer during fine-tuning.
        # I left it unimplemented since now we do not do fine-tuning.
        raise NotImplementedError


def set_seed(
        seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


@torch.no_grad()
def eval_actor(
        env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int
) -> np.ndarray:
    env.reset(seed=seed) # TODO: should we set it in a specific way?
    actor.eval()
    episode_rewards = []
    # one agent is evaluated for n_episodes times
    for _ in range(n_episodes):
        state = env.reset()
        terminated, truncated = False, False
        # State is tuple from simulator_env
        state = state[0]
        episode_reward = 0.0
        while not terminated and not truncated:
            action = actor.act(state, device)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards)


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action):
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

        self.max_action = torch.tensor(max_action, dtype=torch.float32, device=TrainConfig.device)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = TrainConfig.device, deterministic: bool = True) -> np.ndarray:
        state = torch.tensor(state, device=device, dtype=torch.float32)
        return self(state).cpu().data.numpy().flatten()


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(Critic, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], 1)
        return self.net(sa)


class TD3_BC:
    def __init__(
            self,
            max_action: float,
            actor: nn.Module,
            actor_optimizer: torch.optim.Optimizer,
            critic_1: nn.Module,
            critic_1_optimizer: torch.optim.Optimizer,
            critic_2: nn.Module,
            critic_2_optimizer: torch.optim.Optimizer,
            discount: float = 0.99,
            tau: float = 0.005,
            policy_noise: float = 0.2,
            noise_clip: float = 0.5,
            policy_freq: int = 2,
            alpha: float = 2.5,
            grad_clip: float = 2.0,
            device: str = TrainConfig.device
    ):
        self.actor = actor.to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = actor_optimizer
        self.critic_1 = critic_1.to(device)
        self.critic_1_target = copy.deepcopy(self.critic_1)
        self.critic_1_optimizer = critic_1_optimizer
        self.critic_2 = critic_2.to(device)
        self.critic_2_target = copy.deepcopy(self.critic_2)
        self.critic_2_optimizer = critic_2_optimizer

        self.max_action = torch.tensor(max_action, dtype=torch.float, device=device)
        self.discount = discount
        self.tau = tau
        self.policy_noise = torch.tensor(policy_noise, dtype=torch.float, device=device)
        self.noise_clip = torch.tensor(noise_clip, dtype=torch.float, device=device)
        self.policy_freq = policy_freq
        self.alpha = alpha

        self.total_it = 0
        self.device = device
        self.grad_clip = grad_clip

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        log_dict = {}

        state, action, reward, next_state, done = batch
        not_done = 1 - done

        # Create masks for valid states
        state_mask = (state != MISSING_NEARBY_AGENT_VALUE)
        next_state_mask = (next_state != MISSING_NEARBY_AGENT_VALUE)

        with torch.no_grad():
            # Select action according to actor and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )

            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )

            # Compute the target Q value
            target_q1 = self.critic_1_target(next_state * next_state_mask, next_action)
            target_q2 = self.critic_2_target(next_state * next_state_mask, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + not_done * self.discount * target_q

        # Get current Q estimates
        current_q1 = self.critic_1(state * state_mask, action)
        current_q2 = self.critic_2(state * state_mask, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        log_dict["critic_loss"] = critic_loss.item()
        # Optimize the critic
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), self.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), self.grad_clip)
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # Delayed actor updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            pi = self.actor(state * state_mask)
            q = self.critic_1(state * state_mask, pi)
            # using detach to prevent lmbda from affecting q
            lmbda = self.alpha / q.abs().mean().detach()

            actor_loss = -lmbda * q.mean() + F.mse_loss(pi, action)
            log_dict["actor_loss"] = actor_loss.item()
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            # gradually blending the main network's weights into the target network's weights
            soft_update(self.critic_1_target, self.critic_1, self.tau)
            soft_update(self.critic_2_target, self.critic_2, self.tau)
            soft_update(self.actor_target, self.actor, self.tau)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "critic_1": self.critic_1.state_dict(),
            "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.critic_1.load_state_dict(state_dict["critic_1"])
        self.critic_1_optimizer.load_state_dict(state_dict["critic_1_optimizer"])
        self.critic_1_target = copy.deepcopy(self.critic_1)

        self.critic_2.load_state_dict(state_dict["critic_2"])
        self.critic_2_optimizer.load_state_dict(state_dict["critic_2_optimizer"])
        self.critic_2_target = copy.deepcopy(self.critic_2)

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_target = copy.deepcopy(self.actor)

        self.total_it = state_dict["total_it"]


def qlearning_dataset(dataset=None):
    """
    Returns datasets formatted for use by standard Q-learning algorithms,
    with observations, actions, next_observations, rewards, and a terminal
    flag.

    Args:
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()

    Returns:
        A dictionary containing keys:
            observations: An N x dim_obs array of observations.
            actions: An N x dim_action array of actions.
            next_observations: An N x dim_obs array of next observations.
            rewards: An N-dim float array of rewards.
            terminals: An N-dim boolean array of "done" or episode termination flags.
    """
    N = len(dataset.rewards)
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []

    for i in range(N - 1):
        obs = dataset.observations[i].astype(np.float32)
        new_obs = dataset.observations[i + 1].astype(np.float32)
        action = dataset.actions[i].astype(np.float32)
        reward = dataset.rewards[i].astype(np.float32)
        done_bool = bool(dataset.terminals[i])

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)

    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
    }


def evaluate(config, env, actor, trainer, evaluations, ref_max_score, ref_min_score):
    """evaluate the policy at certain evaluation frequency"""
    # Evaluate episode
    eval_scores = eval_actor(
        env,
        actor,
        device=config.device,
        n_episodes=config.n_episodes,
        seed=config.seed,
    )
    eval_score = eval_scores.mean()
    normalized_eval_score = get_normalized_score(eval_score, ref_max_score, ref_min_score) * 100.0
    evaluations.append(normalized_eval_score)

    logger.info(f"Evaluation over {config.n_episodes} episodes: ")
    logger.info(f"{eval_score:.3f} , Agent score: {normalized_eval_score:.3f}")

    if config.checkpoints_path is not None:
        torch.save(
            trainer.state_dict(),
            os.path.join(config.checkpoints_path, f"{TrainConfig.driving_style}_checkpoint.pt"),
        )

    wandb.log(
        {"normalized_score": normalized_eval_score},
        step=trainer.total_it,
    )


def get_normalized_score(score, ref_max_score, ref_min_score):
    if (ref_max_score is None) or (ref_min_score is None):
        raise ValueError("Reference score not provided for env")
    return (score - ref_min_score) / (ref_max_score - ref_min_score)


def load_demonstration_data(driving_style: str, map_name: str, dataset_split: str = "train"):
    """load demonstration data"""
    with open(f'scenarios/data/{dataset_split}/{driving_style}{map_name}_demonstration.pkl', 'rb') as file:
        return pickle.load(file)


def compute_normalization_parameters(state_dim, dataset, normalize):
    """Get state mean, std and reward mean and std for normalization"""
    total_count = np.zeros(state_dim)
    mean_obs = np.zeros(state_dim)
    m2 = np.zeros(state_dim)
    all_rewards = []

    dataset_to_use = dataset['General'] if dataset['General'] else dataset['clustered']
    for agent_mdp in dataset_to_use:
        if normalize:
            all_rewards.extend(agent_mdp.rewards)
            # Iterate over each observation
            for obs in agent_mdp.observations:
                # Create a mask for valid observations
                mask = obs != MISSING_NEARBY_AGENT_VALUE

                # Update total count
                total_count[mask] += 1

                # Compute mean and m2 incrementally for valid observations
                delta = obs[mask] - mean_obs[mask]
                mean_obs[mask] += delta / total_count[mask]
                delta2 = obs[mask] - mean_obs[mask]
                m2[mask] += delta * delta2

    if normalize:
        state_mean = mean_obs
        state_std = np.sqrt(m2 / total_count)
        reward_mean = np.mean(all_rewards)
        reward_std = np.std(all_rewards)
    else:
        state_mean, state_std = 0, 1
        reward_mean, reward_std = 0, 1

    return state_mean, state_std, reward_mean, reward_std


def get_normalisation_parameters(driving_style: str, map_name: str, dataset_split: str, state_dim: int):
    """

    :param driving_style:
    :param map_name:
    :param dataset_split:
    :param state_dim: dimension of the state space. e.g., 34 if using an observation space with 34 elements. Otherwise,
                        self.env.observation_space.shape[0]
    :return: state_mean, state_std, reward_mean, reward_std
    """
    demonstrations = load_demonstration_data(driving_style, map_name, dataset_split)
    return compute_normalization_parameters(state_dim, demonstrations, normalize=True)


def initialize_model(state_dim, action_dim, max_action, config):
    actor = Actor(state_dim, action_dim, max_action).to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)

    critic_1 = Critic(state_dim, action_dim).to(config.device)
    critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=3e-4)
    critic_2 = Critic(state_dim, action_dim).to(config.device)
    critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=3e-4)

    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "critic_1": critic_1,
        "critic_1_optimizer": critic_1_optimizer,
        "critic_2": critic_2,
        "critic_2_optimizer": critic_2_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": config.device,
        # TD3
        "policy_noise": config.policy_noise * max_action,
        "noise_clip": config.noise_clip * max_action,
        "policy_freq": config.policy_freq,
        # TD3 + BC
        "alpha": config.alpha,
    }

    trainer = TD3_BC(**kwargs)
    return trainer, actor


class TD3_BC_Loader:
    def __init__(self, config, episode_names, env=None):
        self.config = config

        if env is not None:
            self.env = env
        else:
            self.env = gym.make(config.env, episode_names=episode_names, dataset_split=config.dataset_split,
                                use_irl_reward=config.use_irl_reward,
                                clustering=config.driving_style, spawn_method=config.spawn_method)

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.max_action = self.env.action_space.high

        self.dataset = load_demonstration_data(config.driving_style, config.map_name, dataset_split=config.dataset_split)
        self.state_mean, self.state_std, self.reward_mean, self.reward_std = compute_normalization_parameters(
            self.state_dim, self.dataset, config.normalize)

        # Set seeds
        self.set_seed(config.seed)
        self.trainer, self.actor = initialize_model(self.state_dim, self.action_dim, self.max_action, config)

        if config.load_model:
            self.load_model(config.load_model)

        if config.checkpoints_path:
            self.save_checkpoints()

    def set_seed(self, seed: int):
        set_seed(seed, self.env)

    def load_model(self, model_path: str):
        policy_file = Path(model_path)
        self.trainer.load_state_dict(torch.load(policy_file, map_location=torch.device('cpu')))
        self.actor = self.trainer.actor
        logger.info(f"Loaded model from {model_path}")

    def save_checkpoints(self):
        logger.info(f"Checkpoints path: {self.config.checkpoints_path}")
        os.makedirs(self.config.checkpoints_path, exist_ok=True)
        with open(os.path.join(self.config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(self.config, f)

    def get_trainer(self):
        return self.trainer

    def get_actor(self):
        return self.actor


@pyrallis.wrap()
def train(config: TrainConfig):
    logger.info(f"Training TD3 + BC, Env: {config.env}")

    map_configs = ScenarioConfig.load(get_config_path(TrainConfig.map_name))
    idx = map_configs.dataset_split[TrainConfig.dataset_split]
    training_episodes = [x.recording_id for i, x in enumerate(map_configs.episodes) if i in idx]

    trainer_loader = TD3_BC_Loader(config, training_episodes)
    trainer = trainer_loader.get_trainer()
    actor = trainer_loader.get_actor()

    wandb_init(asdict(config))

    env = wrap_env(trainer_loader.env, state_mean=trainer_loader.state_mean, state_std=trainer_loader.state_std,
                   reward_mean=trainer_loader.reward_mean, reward_std=trainer_loader.reward_std,
                   reward_normalization=config.normalize_reward)

    ref_max_score = -float('inf')
    ref_min_score = float('inf')
    # create a replay buffer for each vehicle
    replay_buffers = []
    for agent_mdp in trainer_loader.dataset['clustered']:
        agent_data = qlearning_dataset(dataset=agent_mdp)

        # observation normalization
        agent_data["observations"] = normalize_states(
            agent_data["observations"], trainer_loader.state_mean, trainer_loader.state_std
        )
        agent_data["next_observations"] = normalize_states(
            agent_data["next_observations"], trainer_loader.state_mean, trainer_loader.state_std
        )

        # reward normalization
        agent_data["rewards"] = normalized_rewards(
            agent_data["rewards"], trainer_loader.reward_mean, trainer_loader.reward_std
        )

        score = sum(agent_data["rewards"])
        if score > ref_max_score:
            ref_max_score = score
        if score < ref_min_score:
            ref_min_score = score

        replay_buffer = ReplayBuffer(
            trainer_loader.state_dim,
            trainer_loader.action_dim,
            config.buffer_size,
            config.device,
        )
        replay_buffer.load_automatum_dataset(agent_data)
        replay_buffers.append(replay_buffer)

    # Training loop
    logger.info("Training on data from each agent separately!")
    evaluations = []
    for t in tqdm(range(int(config.max_timesteps))):
        iteration_log_dict = {}
        for replay_buffer in replay_buffers:
            batch = replay_buffer.sample(config.batch_size)
            batch = [b.to(config.device) for b in batch]
            log_dict = trainer.train(batch)

            # Accumulate logs from this batch into the iteration log dictionary
            for key, value in log_dict.items():
                if key in iteration_log_dict:
                    iteration_log_dict[key].append(value)
                else:
                    iteration_log_dict[key] = [value]
        trainer.total_it += 1

        # Calculate the average of the accumulated logs for this iteration
        averaged_log_dict = {key: sum(values) / len(values) for key, values in iteration_log_dict.items()}
        wandb.log(averaged_log_dict, step=trainer.total_it)

        # Evaluate episode
        if (t + 1) % config.eval_freq == 0:
            logger.info(f'evaluate at time step: {t + 1}')
            # evaluate the policy
            evaluate(config, env, actor, trainer, evaluations, ref_max_score, ref_min_score)


if __name__ == "__main__":
    train()

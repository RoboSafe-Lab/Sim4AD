# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass
from typing import List

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
import stable_baselines3 as sb3
import wandb

import gym_env  # If this fails, install it with `pip install -e .` from \simulator\gym_env

import logging

from sim4ad.offlinerlenv.td3bc_automatum import wrap_env, TD3_BC_Loader, get_normalisation_parameters

logging.basicConfig(level=logging.INFO)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "SimulatorEnv-v0"  # "Hopper-v4"
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    hidden_layer_dim = 256
    """the hidden layer dimension of (all) the networks"""

    cluster: str = "All"
    """the clustering method to use. Options include 'All', 'Aggressive', 'Normal', 'Cautious'"""

    normalize_state: bool = True
    """whether to normalise the state and the reward"""

    use_irl_reward: bool = False
    """whether to use the IRL reward or the basic default reward"""

    evaluation_seeds: List[int] = (0, 1, 2, 3, 4)  # TODO: change this to 5 different seeds -- currently not used


def make_env(env_id, seed, run_name, args, evaluation=False, normalisation: bool = False):

    logging.info(f"[SAC] Using IRL reward: {args.use_irl_reward}")
    logging.info(f"[SAC] Using cluster: {args.cluster}")

    if evaluation:
        env = gym.make(env_id, dataset_split="valid", use_irl_reward=args.use_irl_reward, clustering=args.cluster,
                       spawn_method="dataset_one")
    else:
        env = gym.make(env_id, use_irl_reward=args.use_irl_reward, clustering=args.cluster, spawn_method="dataset_one")

    if normalisation:
        state_mean, state_std, reward_mean, reward_std = get_normalisation_parameters(driving_style=env.unwrapped.driving_style,
                                                                                      map_name=env.unwrapped.map_name)
        env = wrap_env(env, state_mean=state_mean, state_std=state_std, reward_mean=reward_mean, reward_std=reward_std,
                       reward_normalization=True)

    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(seed)
    return env


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    ### TODO: this predicts the soft Q value function given a state and action
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod() + np.prod(env.action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env, device):
        super().__init__()
        self.device = device
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.name = "SACActor"

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, obs):

        if not isinstance(obs, torch.Tensor):
            obs = torch.Tensor(obs).to(self.device)

        # Given the observation, find the mean and log_std of the action distribution
        mean, log_std = self(obs)
        std = log_std.exp()
        # Initialise a normal distribution with `mean` and `std`
        normal = torch.distributions.Normal(mean, std)
        # Sample with the `r`eparameterization trick to allow backpropagation
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        # Compute the log probability of the sampled action (x_t) under the `normal` distribution.
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum()  # We want a scalar for the log probability of the different dimensions of the action
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    @torch.no_grad()
    def act(self, obs, device=None, deterministic=False):
        """TODO: Device is not used, but added for common interface!"""
        assert device == self.device, f"The device is not matching! {device} != {self.device}"

        if not isinstance(obs, torch.Tensor):
            obs = torch.Tensor(obs).to(self.device)

        action, log_prob, mean = self.get_action(obs)

        if deterministic:
            # Return the most likely action
            return mean.cpu().data.numpy().flatten()

        # otherwise, use the action that was sampled from the distribution
        return action.cpu().data.numpy().flatten()


def evaluate(evaluation_seeds, actor, eval_env, device):
    actor.eval()
    all_test_rets = []
    for seed in evaluation_seeds:
        obs, _ = eval_env.reset(seed=seed)  # TODO: currently the seed may not do anything (?)
        episodic_return = 0
        while True:
            action = actor.act(torch.Tensor(obs).to(device), device=device)
            next_obs, reward, termination, truncation, info = eval_env.step(action)
            episodic_return += reward
            obs = next_obs
            if termination or truncation:
                all_test_rets.append(episodic_return)
                break
    wandb.log({"charts/eval_return": np.mean(all_test_rets)})
    actor.train()
    return np.mean(all_test_rets)
"""
def load_td3bc_checkpoint(checkpoint_path, actor, qf1, qf2, qf1_target, qf2_target, device):
    # load checkpoint.pt
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    #  'actor', 'critic1', 'critic2'
    if 'actor' in checkpoint:
        actor.load_state_dict(checkpoint['actor'])
        logging.info("Loaded actor weights from TD3+BC checkpoint.")
    else:
        raise KeyError("Checkpoint does not contain 'actor' key.")
    
    if 'critic1' in checkpoint and 'critic2' in checkpoint:
        qf1.load_state_dict(checkpoint['critic1'])
        qf2.load_state_dict(checkpoint['critic2'])
        logging.info("Loaded critic weights from TD3+BC checkpoint.")
    else:
        raise KeyError("Checkpoint does not contain 'critic1' and 'critic2' keys.")
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    logging.info("target network follow")
"""

import torch
import logging

def load_td3bc_to_sac(checkpoint_path, sac_actor, qf1, qf2):
    """
     TD3+BC load Actor and Critic to SAC Actor and Soft Q Networks。

    Args:
        checkpoint_path (str): TD3+BC 
        sac_actor (SACActor): SAC 的 Actor
        qf1 (SoftQNetwork): SAC Soft Q Network 1
        qf2 (SoftQNetwork): SAC Soft Q Network 2
    """
    # load
    td3bc_checkpoint = torch.load(checkpoint_path, map_location='cpu') 

    if 'actor' in td3bc_checkpoint:
        td3bc_actor_state_dict = td3bc_checkpoint['actor']
        sac_actor_state_dict = sac_actor.state_dict()

        # TD3+BC net.0 net.2  SAC  fc1 fc2
        mapping_actor = {
            'net.0.weight': 'fc1.weight',
            'net.0.bias': 'fc1.bias',
            'net.2.weight': 'fc2.weight',
            'net.2.bias': 'fc2.bias',
        }

        for td3_key, sac_key in mapping_actor.items():
            if td3_key in td3bc_actor_state_dict and sac_key in sac_actor_state_dict:
                sac_actor_state_dict[sac_key] = td3bc_actor_state_dict[td3_key]
                logging.info(f"Loaded {td3_key} into {sac_key}")
            else:
                logging.warning(f"Key '{td3_key}' or '{sac_key}' not found in state_dict.")

        sac_actor.load_state_dict(sac_actor_state_dict)
    else:
        raise KeyError("Checkpoint does not contain 'actor' key.")

    #  Critic
    if 'critic_1' in td3bc_checkpoint and 'critic_2' in td3bc_checkpoint:
        # TD3+BC net.0, net.2, net.4  SAC  fc1, fc2, fc3
        mapping_critic = {
            'net.0.weight': 'fc1.weight',
            'net.0.bias': 'fc1.bias',
            'net.2.weight': 'fc2.weight',
            'net.2.bias': 'fc2.bias',
            'net.4.weight': 'fc3.weight',
            'net.4.bias': 'fc3.bias',
        }

        #  critic_1 to qf1
        td3bc_critic1_state_dict = td3bc_checkpoint['critic_1']
        sac_qf1_state_dict = qf1.state_dict()

        for td3_key, sac_key in mapping_critic.items():
            if td3_key in td3bc_critic1_state_dict and sac_key in sac_qf1_state_dict:
                sac_qf1_state_dict[sac_key] = td3bc_critic1_state_dict[td3_key]
                logging.info(f"Loaded {td3_key} into qf1.{sac_key}")
            else:
                logging.warning(f"Key '{td3_key}' or 'qf1.{sac_key}' not found in state_dict.")

        # state_dict  qf1
        qf1.load_state_dict(sac_qf1_state_dict)

        # critic_2  qf2
        td3bc_critic2_state_dict = td3bc_checkpoint['critic_2']
        sac_qf2_state_dict = qf2.state_dict()

        for td3_key, sac_key in mapping_critic.items():
            if td3_key in td3bc_critic2_state_dict and sac_key in sac_qf2_state_dict:
                sac_qf2_state_dict[sac_key] = td3bc_critic2_state_dict[td3_key]
                logging.info(f"Loaded {td3_key} into qf2.{sac_key}")
            else:
                logging.warning(f"Key '{td3_key}' or 'qf2.{sac_key}' not found in state_dict.")

        #  state_dict qf2
        qf2.load_state_dict(sac_qf2_state_dict)
    else:
        raise KeyError("Checkpoint does not contain both 'critic_1' and 'critic_2' keys.")


def print_checkpoint_keys(checkpoint_path):

    checkpoint = torch.load(checkpoint_path, map_location='cpu')  

    print("Checkpoint Keys:")
    for key in checkpoint.keys():
        print(f"- {key}")

    for key in ['actor', 'critic_1', 'critic_2']:
        if key in checkpoint:
            print(f"\nKeys in '{key}':")
            state_dict = checkpoint[key]
            for sub_key in state_dict.keys():
                print(f"  - {sub_key}")
        else:
            print(f"\nWarning: '{key}' not found in the checkpoint.")

def main():
    
    CHECKPOINT_PATH = "D:/IRLcode/Sim4AD/results/offlineRL/Normal_checkpoint.pt" # load td3+bc checkpoint
    CHECKPOINT_PATH = "/users/yx3006/Sim4AD/results/offlineRL/Normal_checkpoint.pt"
    #print_checkpoint_keys(CHECKPOINT_PATH)
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    import wandb

    wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        sync_tensorboard=False,
        config=vars(args),
        name=run_name,
        monitor_gym=True,
        save_code=True,
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    env = make_env(args.env_id, seed=args.seed, args=args, run_name=run_name, normalisation=args.normalize_state)
    assert isinstance(env.action_space, gym.spaces.Box), "only continuous action space is supported"

    eval_env = make_env(args.env_id, seed=args.seed, args=args, run_name=run_name, evaluation=True,
                        normalisation=args.normalize_state)

    max_action = float(env.action_space.high[0])

    actor = Actor(env, device=device).to(device)
    qf1 = SoftQNetwork(env).to(device)
    qf2 = SoftQNetwork(env).to(device)
    qf1_target = SoftQNetwork(env).to(device)
    qf2_target = SoftQNetwork(env).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    env.observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        env.observation_space,
        env.action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    try:
        load_td3bc_to_sac(CHECKPOINT_PATH, actor, qf1, qf2)
    except KeyError as e:
        logging.error(f"defeat: {e}")
    
    # update target network
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    # TRY NOT TO MODIFY: start the game
    obs, _ = env.reset(seed=args.seed)
    episodic_return = 0
    episodic_length = 0
    best_eval = -1e6
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            action = env.action_space.sample()
        else:
            action = actor.act(torch.Tensor(obs).to(device), device=device)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, termination, truncation, info = env.step(action)
        episodic_return += reward

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        # TODO!
        if termination or truncation:
            print(f"global_step={global_step}, episodic_return={episodic_return}")
            wandb.log({"charts/episodic_return": episodic_return,
                       "charts/episodic_length": episodic_length})
            episodic_return = 0
            episodic_length = 0
        else:
            episodic_length += 1

        if termination:
            # if the environment is done, the next_obs is nan; make it a goal state
            next_obs = np.ones_like(next_obs) * 2  # TODO: Does it make sense?
        elif truncation:
            next_obs = np.ones_like(next_obs) * 3

        rb.add(np.array([obs]), np.array([next_obs]), np.array([action]), np.array([reward]), np.array([termination]),
               [info])

        # TODO: need to reset env!!!
        if termination or truncation:
            obs, _ = env.reset(seed=args.seed)
        else:
            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)

            with torch.no_grad():
                # The target Q networks use s' while the current Q networks (below) use s
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (
                    min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                        args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                wandb.log({"losses/qf1_values": qf1_a_values.mean().item(),
                           "losses/qf2_values": qf2_a_values.mean().item(),
                           "losses/qf1_loss": qf1_loss.item(),
                           "losses/qf2_loss": qf2_loss.item(),
                           "losses/qf_loss": qf_loss.item() / 2.0,
                           "losses/actor_loss": actor_loss.item(),
                           "losses/alpha": alpha,
                           "charts/SPS": int(global_step / (time.time() - start_time))})

                print("SPS:", int(global_step / (time.time() - start_time)))
                if args.autotune:
                    wandb.log({"losses/alpha_loss": alpha_loss.item()})

            # Evaluate the agent on 5 different seeds
            if global_step % 1000 == 0:
                eval_return = evaluate(args.evaluation_seeds, actor, eval_env, device)
                if eval_return > best_eval:
                    best_eval = eval_return
                    print("Saving new best model")
                    torch.save(actor.state_dict(),
                               f"best_model_sac_{args.cluster}_irl{args.use_irl_reward}_{run_name}.pth")

    env.close()
    eval_env.close()


if __name__ == "__main__":


    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )

    main()

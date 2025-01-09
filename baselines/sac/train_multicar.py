import os
import random
import time
from dataclasses import dataclass
from typing import List, Dict
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
import wandb
import gymnasium as gym
# multi agent environment MultiCarEnv
import sys
sys.path.append("D:\IRLcode\Sim4AD")
from simulator.gym_env.gym_env.envs.multi_simulator_env import MultiCarEnv  # PettingZoo Environment
from stable_baselines3.common.buffers import ReplayBuffer  # 重用SB3的ReplayBuffer?
import logging

logging.basicConfig(level=logging.INFO)

@dataclass
class Args:
    exp_name: str = "MultiAgentSAC"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "multi_sac"
    wandb_entity: str = None
    capture_video: bool = False

    total_timesteps: int = 1000000
    buffer_size: int = int(1e5)
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    learning_starts: int = 5e3
    policy_lr: float = 3e-4
    q_lr: float = 1e-3
    policy_frequency: int = 2
    target_network_frequency: int = 1
    alpha: float = 0.2
    autotune: bool = True
    hidden_layer_dim = 256
    normalize_state: bool = True
    use_irl_reward: bool = False

    # MultiCarEnv related?
    spawn_method: str = "dataset_all"  # or "random"
    clustering: str = "Cautious"
    episode_names = []   
    max_steps: int = 1000

    evaluation_seeds: List[int] = (0,1,2,3,4)

# ----------------------
# SAC Network
# ----------------------
class SoftQNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + act_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        z = torch.cat([x, a], dim=1)
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        q = self.fc3(z)
        return q

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, action_low, action_high, hidden_dim=256, device='cpu'):
        super().__init__()
        self.device = device
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, act_dim)
        self.fc_logstd = nn.Linear(hidden_dim, act_dim)

        # rescaling
        action_scale = (action_high - action_low)/2.
        action_bias = (action_high + action_low)/2.
        self.register_buffer("action_scale", torch.tensor(action_scale, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor(action_bias, dtype=torch.float32))

        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -5

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5*(self.LOG_STD_MAX - self.LOG_STD_MIN)*(log_std + 1)
        return mean, log_std

    def get_action(self, obs):
        mean, log_std = self(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale*(1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        mean_action = torch.tanh(mean)*self.action_scale + self.action_bias
        return action, log_prob, mean_action

    @torch.no_grad()
    def act(self, obs_np, deterministic=False):
        obs_tensor = torch.tensor(obs_np, dtype=torch.float32, device=self.device)
        mean, log_std = self(obs_tensor)
        std = log_std.exp()
        if deterministic:
            y_t = torch.tanh(mean)
        else:
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()
            y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        return action.cpu().numpy()

# multi agent evaluate

def evaluate_multi(env, actor, device, n_eval_episodes=1):
    actor.eval()
    returns = []
    for _ in range(n_eval_episodes):
        obs_dict = env.reset()
        #episode_rewards = {agent_id:0.0 for agent_id in env.agents}
        episode_rewards = {}
        done_all = False
        while not done_all:
            actions_dict = {}
            for agent_id, obs in obs_dict.items():
                if isinstance(actor, nn.DataParallel):
                    action = actor.module.act(obs, deterministic=True)
                else:
                    action = actor.act(obs, deterministic=True)
                actions_dict[agent_id] = action
            next_obs_dict, rewards_dict, dones_dict, infos_dict = env.step(actions_dict)

            #for agent_id in obs_dict:
            for agent_id, reward in rewards_dict.items():
                if agent_id not in episode_rewards:
                    episode_rewards[agent_id] = 0.0
                #episode_rewards[agent_id] += rewards_dict[agent_id]
                episode_rewards[agent_id] += reward

            done_all = dones_dict["__any__"]
            obs_dict = {agent_id: next_obs_dict[agent_id] for agent_id in next_obs_dict if not dones_dict[agent_id]}
        # sum return, all agent average:
        returns.append(np.mean(list(episode_rewards.values())))
    actor.train()
    return np.mean(returns)

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
    #CHECKPOINT_PATH = "/users/yx3006/Sim4AD/results/offlineRL/Cautious_checkpoint.pt" # load td3+bc checkpoint
    CHECKPOINT_PATH = "D:/IRLcode/Sim4AD/results/offlineRL/Normal_checkpoint.pt"
    args = tyro.cli(Args)
    run_name = f"MultiAgentSAC__{args.seed}__{int(time.time())}"
    
    import wandb
    wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        sync_tensorboard=False,
        config=vars(args),
        name=run_name,
        save_code=True,
    )

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # build multi agent environment (based on PettingZoo)
    
    env = MultiCarEnv(
        episode_names=args.episode_names,
        spawn_method=args.spawn_method,
        clustering=args.clustering,
        max_steps=args.max_steps,
    )
    # build eval_env
    eval_env = MultiCarEnv(
        episode_names=args.episode_names,
        spawn_method=args.spawn_method,
        clustering=args.clustering,
        max_steps=args.max_steps,
        evaluation=True
    )

    #obs_dict = env.reset()
    # get observation/ action
    # agents have obs_dim act_dim same
    obs_dim = env._observation_space.shape[0]  # PettingZoo 并行: same space for all
    act_dim = env._action_space.shape[0]
    action_low = env._action_space.low
    action_high = env._action_space.high

    #  Actor & Q
    actor = Actor(obs_dim, act_dim, action_low, action_high, hidden_dim=args.hidden_layer_dim, device=device).to(device)
    qf1 = SoftQNetwork(obs_dim, act_dim, hidden_dim=args.hidden_layer_dim).to(device)
    qf2 = SoftQNetwork(obs_dim, act_dim, hidden_dim=args.hidden_layer_dim).to(device)
    qf1_target = SoftQNetwork(obs_dim, act_dim, hidden_dim=args.hidden_layer_dim).to(device)
    qf2_target = SoftQNetwork(obs_dim, act_dim, hidden_dim=args.hidden_layer_dim).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    try:
        load_td3bc_to_sac(CHECKPOINT_PATH, actor, qf1, qf2)
    except KeyError as e:
        logging.error(f"defeat: {e}")
        
        # 使用 DataParallel 封装模型
    if torch.cuda.device_count() > 1:
        logging.info(f"use {torch.cuda.device_count()}  GPU for training")
        actor = nn.DataParallel(actor)
        qf1 = nn.DataParallel(qf1)
        qf2 = nn.DataParallel(qf2)
        qf1_target = nn.DataParallel(qf1_target)
        qf2_target = nn.DataParallel(qf2_target)

    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    if args.autotune:
        target_entropy = -act_dim
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    # ReplayBuffer
    from stable_baselines3.common.buffers import ReplayBuffer
    rb = ReplayBuffer(
        args.buffer_size,
        gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,)), 
        gym.spaces.Box(low=-np.inf, high=np.inf, shape=(act_dim,)), 
        device=device,
        handle_timeout_termination=False,
    )
    
    # update target network
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    global_step = 0
    start_time = time.time()
    best_eval = -1e9
    episodic_returns = {agent_id:0.0 for agent_id in env.agents}
    episodic_length = 0

    #while global_step < args.total_timesteps:
    for iteration in range(3000):
        logging.info(f"==== Start iteration {iteration} ====")
        obs_dict = env.reset()
        is_episodes_done = False
        env.simulation.reset_done_full_cycle()
        while not is_episodes_done:
            actions_dict = {}
            # all agents 
            for agent_id, obs in obs_dict.items():
                if global_step < args.learning_starts:
                    # random
                    action = env.action_space.sample() #here is a problem no sample function
                else:
                    #action = actor.act(obs, deterministic=False)
                    with torch.no_grad():
                        if torch.cuda.device_count() > 1:
                            action = actor.module.act(obs, deterministic=False)  # 使用 DataParallel 封装后的模块
                        else:
                            action = actor.act(obs, deterministic=False)
                actions_dict[agent_id] = action

            next_obs_dict, rewards_dict, dones_dict, infos_dict = env.step(actions_dict)
            global_step += 1
            episodic_length += 1
            if global_step % 1000 == 0:
                logging.info(f"global_step_up={global_step}, episodic_length_up={episodic_length}, simulation_time={env.current_time()}")
            # all agent experience save in the same ReplayBuffer
            if len(next_obs_dict.keys()) < len(obs_dict.keys()):
                logging.info("there are something wrong in obs!")
            for agent_id in obs_dict.keys():
                #if agent_id not in next_obs_dict.keys():
                    #continue
                try :
                    done = dones_dict[agent_id]
                except KeyError as e :
                    logging.info(", ".join(obs_dict.keys()))
                    logging.info(", ".join(next_obs_dict.keys()))
                    logging.info(f"simulation_time={env.current_time()}")
                    logging.info(", ".join(dones_dict.keys()))
                    
                rew = rewards_dict[agent_id]
                if agent_id not in episodic_returns:
                    episodic_returns[agent_id] = 0.0
                episodic_returns[agent_id] += rew
                old_obs = obs_dict[agent_id]
                action = actions_dict[agent_id]
                new_obs = next_obs_dict[agent_id] if not done else np.zeros(obs_dim, dtype=np.float32)
                rb.add(
                    old_obs, new_obs, action, np.array([rew]), np.array([done]),
                    [infos_dict[agent_id]]
                )

            # remove done的agent
            obs_dict = {agent_id: next_obs_dict[agent_id] for agent_id in next_obs_dict if not dones_dict[agent_id]}
            if not obs_dict:
                logging.info("the reset obs_dict is None something error")


            # if__all__ done，reset
            if dones_dict["__any__"] or abs(env.current_time() - env.end_time()) <= env.step_time() :
                # all agent return
                mean_return = np.mean(list(episodic_returns.values()))
                logging.info(f"global_step={global_step}, episodic_return={mean_return}, episodic_length={episodic_length}")
                wandb.log({"charts/episodic_return": mean_return,
                        "charts/episodic_length": episodic_length}, step=global_step)
                episodic_returns = {agent_id:0.0 for agent_id in env.agents}
                episodic_length = 0
                # all episodes are tranversed
                if env.is_done_full_cycle() :
                    is_episodes_done = True
                if not env.is_done_full_cycle() :
                    obs_dict = env.reset()

            # SAC update
            if global_step > args.learning_starts:
                data = rb.sample(args.batch_size)
                # data.observations shape (batch, obs_dim)
                # data.actions shape (batch, act_dim)
                with torch.no_grad():
                    if torch.cuda.device_count() > 1:
                        next_state_actions, next_state_log_pi, _ = actor.module.get_action(data.next_observations)
                    else:
                        next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                    qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                    qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * min_qf_next_target.view(-1)

                qf1_a_values = qf1(data.observations, data.actions).view(-1)
                qf2_a_values = qf2(data.observations, data.actions).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                if global_step % args.policy_frequency == 0:
                    for _ in range(args.policy_frequency):
                        pi, log_pi, _ = actor.get_action(data.observations)
                        qf1_pi = qf1(data.observations, pi)
                        qf2_pi = qf2(data.observations, pi)
                        min_qf_pi = torch.min(qf1_pi, qf2_pi)
                        actor_loss = (alpha * log_pi - min_qf_pi).mean()

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

                if global_step % args.target_network_frequency == 0:
                    for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau)*target_param.data)
                    for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau)*target_param.data)


      
        wandb.log({
            "losses/qf_loss": qf_loss.item()/2,
            "losses/qf1_loss": qf1_loss.item(),
            "losses/qf2_loss": qf2_loss.item(),
            "alpha": alpha,
        }, step=global_step)

        # eval
        if iteration % 10 == 0 :
            eval_return = evaluate_multi(eval_env, actor, device, n_eval_episodes=1)
            wandb.log({"charts/eval_return": eval_return}, step=global_step)
            if eval_return > best_eval:
                logging.info("model with better return is found")
                best_eval = eval_return
                if isinstance(actor, nn.DataParallel):
                    torch.save(actor.module.state_dict(), f"best_model_sac_multi.pth")
                else:
                    torch.save(actor.state_dict(), f"best_model_sac_multi.pth")

    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()

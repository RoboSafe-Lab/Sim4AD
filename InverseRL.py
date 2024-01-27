import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pickle
import random


# Assuming you have a PyTorch model for your policy
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        # Define the architecture here
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, state):
        # Implement the forward pass
        return self.fc(state)


class RewardNetwork(nn.Module):
    def __init__(self, state_dim):
        super(RewardNetwork, self).__init__()
        # Define the architecture here
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        # Implement the forward pass
        return self.fc(state)


def compute_policy_loss(policy_network, states, actions):
    """
    Computes policy loss for a continuous action space.

    Args:
    policy_network (nn.Module): The policy network.
    states (torch.Tensor): The states observed.
    actions (torch.Tensor): The actions taken by the expert.

    Returns:
    torch.Tensor: The computed loss.
    """
    # Assuming the policy network outputs mean and log_std for each action
    mean, log_std = policy_network(states)
    std = torch.exp(log_std)

    # Create a normal distribution with the predicted mean and std
    normal_dist = torch.distributions.Normal(mean, std)

    # Compute the log probability of the taken actions
    log_probs = normal_dist.log_prob(actions)

    # The loss is the negative log probability
    loss = -log_probs.mean()

    return loss


def compute_irl_loss(reward_network, policy_network, expert_states, expert_actions, num_samples=10):
    """
    Compute IRL loss. This is a simplified version and should be adapted
    based on your exact approach and environment.
    """
    expert_rewards = reward_network(expert_states).mean()
    sampled_rewards = []

    for _ in range(num_samples):
        # Sample some states and actions (this is environment-specific)
        sampled_states, sampled_actions = sample_states_and_actions(expert_trajectories, num_samples)
        sampled_states = torch.tensor(sampled_states, dtype=torch.float32)
        sampled_actions = torch.tensor(sampled_actions, dtype=torch.float32)

        # Compute the reward for these sampled states
        rewards = reward_network(sampled_states)
        policy_loss = compute_policy_loss(policy_network, sampled_states, sampled_actions)
        sampled_rewards.append(rewards.mean() - policy_loss)

    sampled_rewards = torch.stack(sampled_rewards).mean()

    # The loss aims to increase the rewards for expert states more than the sampled states
    return sampled_rewards - expert_rewards


def sample_states_and_actions(expert_trajectories, num_samples):
    """
    Sample states and actions from expert trajectories.

    Args:
    expert_trajectories (list): List of trajectories, where each trajectory is a list of (state, action) tuples.
    num_samples (int): Number of samples to generate.

    Returns:
    list: Sampled states and actions.
    """
    sampled_states = []
    sampled_actions = []

    # Flatten the list of trajectories into a list of state-action pairs
    all_state_action_pairs = [pair for trajectory in expert_trajectories for pair in trajectory]

    # Randomly sample state-action pairs
    for _ in range(num_samples):
        state, action = random.choice(all_state_action_pairs)
        sampled_states.append(state)
        sampled_actions.append(action)

    return sampled_states, sampled_actions


def train_reward_network(reward_network, policy_network, expert_trajectories, reward_optimizer, policy_optimizer,
                         num_iterations=100, num_policy_updates=10):
    for iteration in range(num_iterations):
        total_irl_loss = 0
        total_policy_loss = 0

        # Update the reward network
        for expert_states, expert_actions in expert_trajectories:
            expert_states = torch.tensor(expert_states, dtype=torch.float32)
            expert_actions = torch.tensor(expert_actions, dtype=torch.float32)

            # Compute IRL Loss
            irl_loss = compute_irl_loss(reward_network, policy_network, expert_states, expert_actions)
            reward_optimizer.zero_grad()
            irl_loss.backward()
            reward_optimizer.step()

            total_irl_loss += irl_loss.item()

        # Update the policy network
        for _ in range(num_policy_updates):
            # Sample states and actions from demonstrations
            sampled_states, sampled_actions = sample_states_and_actions(expert_trajectories, 100)
            sampled_states = torch.tensor(sampled_states, dtype=torch.float32)
            sampled_actions = torch.tensor(sampled_actions, dtype=torch.float32)

            # Compute Policy Loss
            policy_loss = compute_policy_loss(policy_network, sampled_states, sampled_actions)
            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()

            total_policy_loss += policy_loss.item()

        print(
            f"Iteration {iteration + 1}/{num_iterations}, IRL Loss: {total_irl_loss}, Policy Loss: {total_policy_loss}")


def load_demonstrations():
    """Loading the demonstrations for training"""
    demonstrations = []
    data_path = 'scenarios/data/trainingdata'
    folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
    for folder in folders:
        with open(folder, 'rb') as file:
            # Load the contents from the file
            data = pickle.load(file)
        demonstrations.append(data)

    return data


# Load your expert trajectories
# Format: [[(state1, action1), (state2, action2), ...], ...]
expert_trajectories = load_demonstrations()

# Define your environment's state and action dimensions
state_dim = 50
action_dim = 2

# Create networks
reward_network = RewardNetwork(state_dim)
policy_network = PolicyNetwork(state_dim, action_dim)

# Optimizers
reward_optimizer = optim.Adam(reward_network.parameters(), lr=0.01)
policy_optimizer = optim.Adam(policy_network.parameters(), lr=0.01)

# Train the reward network
train_reward_network(reward_network, policy_network, expert_trajectories, reward_optimizer, policy_optimizer)
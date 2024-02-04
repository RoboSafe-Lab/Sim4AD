"""
Behavioural cloning baseline for automatum dataset
"""

import os
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score

# We use a Recurrent Neural Network (RNN) to model the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, loss_function=nn.MSELoss(reduction="none")):
        super(PolicyNetwork, self).__init__()
        # Define the architecture here
        self.rnn = nn.LSTM(input_size=state_dim, hidden_size=128, num_layers=2, batch_first=True)
        self.fc = nn.Linear(128, action_dim)
        self.loss_function = loss_function

    def forward(self, state):

        # out contains the hidden state for each time step of the trajectory, after all the layers
        out, _ = self.rnn(state)
        return self.fc(out) # TODO: check if this is the correct way to get the last output

def compute_policy_loss(policy_network, states, expert_actions):
    """
    Computes policy loss for a continuous action space.

    Args:
    policy_network (nn.Module): The policy network.
    states (torch.Tensor): The states observed.
    actions (torch.Tensor): The actions taken by the expert.

    Returns:
    torch.Tensor: The computed loss.
    """

    # For each state/trajectory, we want to compute the loss of the predicted action with respect to the expert action
    # at each TIMESTEP (i.e. the loss is computed at each timestep and then averaged over all timesteps)

    # I have a batch_size*seq_len*action_dim tensor, where the first dimension is the batch size, the second is the
    # sequence length and the third is the action dimension
    # I have to compare this with the expert actions, which are a batch_size*seq_len*action_dim tensor
    # (e.g., (ax, ay, steering_angle))
    # Get the predicted actions from the policy network
    predicted_actions = policy_network(states)

    # Define the loss function (you can use different loss functions depending on your task)
    criterion = policy_network.loss_function

    # Compute the loss between predicted actions and expert actions
    return criterion(predicted_actions, expert_actions)


def train_policy_network(policy_network, expert_states, expert_actions, num_epochs=100, batch_size=32, lr=1e-3):
    """
    Trains the policy network using behavioural cloning.

    Args:
    policy_network (nn.Module): The policy network.
    expert_states (torch.Tensor): The states observed by the expert.
    expert_actions (torch.Tensor): The actions taken by the expert.
    num_epochs (int): The number of epochs to train the network.
    batch_size (int): The batch size for training.
    lr (float): The learning rate for training.

    Returns:
    nn.Module: The trained policy network.
    """
    # Create a DataLoader for the expert data
    dataset = TensorDataset(expert_states, expert_actions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create an optimizer for the policy network
    optimizer = optim.Adam(policy_network.parameters(), lr=lr)

    # Train the policy network
    for epoch in range(num_epochs):
        for states, actions in dataloader:
            optimizer.zero_grad()
            loss = compute_policy_loss(policy_network, states, actions)
            loss.backward()
            optimizer.step()

    return policy_network


def train_reward_network(reward_network, expert_states, expert_rewards, num_epochs=100, batch_size=32, lr=1e-3):
    """
    Trains the reward network using supervised learning.

    Args:
    reward_network (nn.Module): The reward network.
    expert_states (torch.Tensor): The states observed by the expert.
    expert_rewards (torch.Tensor): The rewards observed by the expert.
    num_epochs (int): The number of epochs to train the network.
    batch_size (int): The batch size for training.
    lr (float): The learning rate for training.

    Returns:
    nn.Module: The trained reward network.
    """
    # Create a DataLoader for the expert data
    dataset = TensorDataset(expert_states, expert_rewards)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create an optimizer for the reward network
    optimizer = optim.Adam(reward_network.parameters(), lr=lr)

    # Train the reward network
    for epoch in range(num_epochs):
        for states, rewards in dataloader:
            optimizer.zero_grad()
            loss = compute_reward_loss(reward_network, states, rewards)
            loss.backward()
            optimizer.step()

    return reward_network


def evaluate_policy(policy_network, states, actions):
    """
    Evaluates the policy network on a dataset.

    Args:
    policy_network (nn.Module): The policy network.
    states (torch.Tensor): The states to evaluate on.
    actions (torch.Tensor): The actions to evaluate on.

    Returns:
    float: The mean squared error of the policy network's predictions.
    """
    # Compute the mean squared error of the policy network's predictions
    with torch.no_grad():
        predicted_actions = policy_network(states)
        return mean_squared_error(actions, predicted_actions)

# TODO: implement this function
def evaluate_reward(reward_network, states, rewards):
    """
    Evaluates the reward network on a dataset.

    Args:
    reward_network (nn.Module): The reward network.
    states (torch.Tensor): The states to evaluate on.
    rewards (torch.Tensor): The rewards to evaluate on.

    Returns:
    float: The mean squared error of the reward network's predictions.
    """
    # Compute the mean squared error of the reward network's predictions
    with torch.no_grad():
        predicted_rewards = reward_network(states)
        return mean_squared_error(rewards, predicted_rewards)


def main():
    # Load the expert data
    with open(
            'scenarios/data/trainingdata/hw-a9-appershofen-001-d8087340-8287-46b6-9612-869b09e68448/demonstration.pkl',
            'rb') as f:
        expert_data = pickle.load(f)

    # TODO: choose a good padding value
    expert_states_all = pad_sequence([torch.as_tensor(seq, dtype=torch.float32) for seq in expert_data['observations']], batch_first=True,
                                        padding_value=-1)
    expert_actions_all = pad_sequence([torch.as_tensor(seq, dtype=torch.float32) for seq in expert_data['actions']], batch_first=True,
                                        padding_value=-1)

    # TODO: in the dataset above there should also be the episode id and the agent id (?)
    # Split the expert data into training and testing sets using pytorch
    expert_states_train, expert_states_test, expert_actions_train, expert_actions_test = train_test_split(
        expert_states_all, expert_actions_all, test_size=0.2)

    # Normalize the states using PyTorch transformations
    normalize = transforms.Normalize(mean=expert_states_train.mean(dim=0),
                                     std=expert_states_train.std(dim=0) + 1e-8) # add a small number to avoid division by zero

    expert_states_train = normalize(expert_states_train)
    expert_states_test = normalize(expert_states_test)

    # Create the policy and reward networks
    policy_network = PolicyNetwork(state_dim=expert_states_train.shape[-1], action_dim=expert_actions_train.shape[-1])

    # Train the policy network
    policy_network = train_policy_network(policy_network, expert_states_train, expert_actions_train)

    # Evaluate the policy network
    policy_mse = evaluate_policy(policy_network, states_test, actions_test)
    print(f'Policy MSE: {policy_mse}')

    # Evaluate the reward network
    reward_mse = evaluate_reward(reward_network, states_test, rewards_test)
    print(f'Reward MSE: {reward_mse}')


if __name__ == '__main__':
    main()

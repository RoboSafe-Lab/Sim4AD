"""
Behavioural cloning baseline for the automatum dataset
"""

import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# We use a Recurrent Neural Network (RNN) to model the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, loss_function=nn.MSELoss(reduction="none")):
        super(PolicyNetwork, self).__init__()

        self.rnn = nn.LSTM(input_size=state_dim, hidden_size=128, num_layers=2, batch_first=True)
        self.fc = nn.Linear(128, action_dim)
        self.loss_function = loss_function

    def forward(self, state):
        # out contains the hidden state for each time step of the trajectory, after all the layers
        out, _ = self.rnn(state)
        assert out.shape[:2] == state.shape[:2], f"out.shape: {out.shape}, state.shape: {state.shape}"

        return self.fc(out)


def compute_policy_loss(policy_network, states, expert_actions):
    """

    Args:
    policy_network (nn.Module): The policy network.
    states (torch.Tensor): The states observed.
    expert_actions (torch.Tensor): The actions taken by the expert.

    Returns:
    torch.Tensor: Difference of the action taken by the expert vs the policy we are training.
    """

    # Get the predicted actions from the policy network
    predicted_actions = policy_network(states)

    # Compute the loss between predicted actions and expert actions. We want to average the loss at each timestep for
    # each trajectory to get a single scalar value.
    loss = policy_network.loss_function(predicted_actions, expert_actions)
    assert loss.shape == expert_actions.shape, f"loss.shape: {loss.shape}, expert_actions.shape: {expert_actions.shape}"

    return loss.mean()


def train_policy_network(policy_network, expert_states, expert_actions, num_epochs=100, batch_size=32,
                         lr=1e-3):  # TODO: change number of epochs
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

    dataset = TensorDataset(expert_states, expert_actions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(policy_network.parameters(), lr=lr)

    # Training loop
    for epoch in tqdm(range(num_epochs)):
        for states, actions in dataloader:
            optimizer.zero_grad()
            loss = compute_policy_loss(policy_network, states, actions)
            loss.backward()
            optimizer.step()

    return policy_network


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
        loss = nn.MSELoss()
        return loss(predicted_actions, actions).item()


def main():
    # Load the expert data
    # TODO: load other episodes as well!
    with open(
            'scenarios/data/trainingdata/hw-a9-appershofen-001-d8087340-8287-46b6-9612-869b09e68448/demonstration.pkl',
            'rb') as f:
        expert_data = pickle.load(f)

    PADDING_VALUE = -1  # used to pad the LSTM input to the same length
    expert_states_all = pad_sequence([torch.as_tensor(seq, dtype=torch.float32) for seq in expert_data['observations']],
                                     batch_first=True,
                                     padding_value=PADDING_VALUE)
    expert_actions_all = pad_sequence([torch.as_tensor(seq, dtype=torch.float32) for seq in expert_data['actions']],
                                      batch_first=True,
                                      padding_value=PADDING_VALUE)

    expert_states_train, expert_states_test, expert_actions_train, expert_actions_test = train_test_split(
        expert_states_all, expert_actions_all, test_size=0.2)

    normalize = transforms.Normalize(mean=expert_states_train.mean(dim=0),
                                     std=expert_states_train.std(
                                         dim=0) + 1e-8)  # add a small number to std to avoid division by zero

    expert_states_train = normalize(expert_states_train)
    expert_states_test = normalize(expert_states_test)

    policy_network = PolicyNetwork(state_dim=expert_states_train.shape[-1], action_dim=expert_actions_train.shape[-1])

    policy_network = train_policy_network(policy_network, expert_states_train, expert_actions_train)

    policy_mse = evaluate_policy(policy_network, expert_states_test, expert_actions_test)
    print(f'Policy MSE: {policy_mse}')


if __name__ == '__main__':
    main()

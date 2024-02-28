"""
Behavioural cloning baseline for the automatum dataset
"""
import logging
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from sim4ad.path_utils import baseline_path

logger = logging.getLogger(__name__)

# TODO: could implement early stopping / saving only best model rather than final one

# We use a Recurrent Neural Network (RNN) to model the policy network
class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()

        # TODO: load other episodes as well!
        expert_data = {"observations": [], "actions": []}
        for scenario in ['hw-a9-appershofen-001-d8087340-8287-46b6-9612-869b09e68448', 'hw-a9-appershofen-002-2234a9ae-2de1-4ad4-9f43-65c2be9696d6',
                         'hw-a9-appershofen-003-6d6e3378-df9b-4130-8cbf-3437a77a309d']:
            with open(f'scenarios/data/trainingdata/{scenario}/demonstration.pkl', 'rb') as f:
                new_expert_data = pickle.load(f)
                expert_data['observations'] += new_expert_data['observations']
                expert_data['actions'] += new_expert_data['actions']

        PADDING_VALUE = -1  # used to pad the LSTM input to the same length
        expert_states_all = pad_sequence(
            [torch.as_tensor(seq, dtype=torch.float32) for seq in expert_data['observations']],
            batch_first=True,
            padding_value=PADDING_VALUE)
        expert_actions_all = pad_sequence([torch.as_tensor(seq, dtype=torch.float32) for seq in expert_data['actions']],
                                          batch_first=True,
                                          padding_value=PADDING_VALUE)

        expert_states_train, expert_states_test, expert_actions_train, expert_actions_test = train_test_split(
            expert_states_all, expert_actions_all, test_size=0.2)

        # TODO: should we normalize the data?
        #normalize = transforms.Normalize(mean=expert_states_train.mean(dim=0),
        #                                 std=expert_states_train.std(
        #                                     dim=0) + 1e-8)  # add a small number to std to avoid division by zero
        #expert_states_train = normalize(expert_states_train)
        #expert_states_test = normalize(expert_states_test)
        expert_actions_train = expert_actions_train
        expert_actions_test = expert_actions_test

        # Discretize the actions to 2 decimal places
        expert_actions_train = torch.round(expert_actions_train * 100) / 100
        expert_actions_test = torch.round(expert_actions_test * 100) / 100

        input_space = expert_states_train.shape[-1]
        action_space = expert_actions_train.shape[-1]

        self.rnn = nn.LSTM(input_size=input_space, hidden_size=256, num_layers=2, batch_first=True)
        self.fc = nn.Linear(256, action_space)

        # TODO: should be 34 as we should not include x/y
        assert input_space == 36 and action_space == 2  # TODO: just for automatum dataset
        self.loss_function = nn.MSELoss(reduction="mean")  # TODO: is this the correct loss function / reduction?
        self.writer = SummaryWriter(f'baselines/runs/bc')

        self.BATCH_SIZE = 64  # Define your batch size # TODO: parameterize
        self.SHUFFLE = True  # shuffle your data
        self.EPOCHS = 500  # Define the number of epochs # TODO: parameterize
        self.LR = 1e-3  # Define your learning rate # TODO: parameterize

        self.train_loader = DataLoader(AutomatumDataset(expert_states_train, expert_actions_train),
                                       batch_size=self.BATCH_SIZE, shuffle=self.SHUFFLE)
        self.eval_loader = DataLoader(AutomatumDataset(expert_states_test, expert_actions_test),
                                      batch_size=self.BATCH_SIZE, shuffle=self.SHUFFLE)

        self.eval_losses = []

        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, history: torch.Tensor):
        # out contains the hidden state for each time step of the trajectory, after all the layers
        if isinstance(history, list):
            history = torch.tensor(history, dtype=torch.float32)
        out, _ = self.rnn(history)

        return self.fc(out)

    def load_policy(self, baseline_name='bc'):
        path = baseline_path(baseline_name)
        self.load_state_dict(torch.load(path))

    def train(self, num_epochs=100, learning_rate=1e-3):
        best_loss = float('inf')

        for epoch in tqdm(range(num_epochs)):
            # Training
            for i, (state, action) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                predicted_actions = self.forward(state)
                loss = self.loss_function(predicted_actions, action)
                loss.backward()
                self.optimizer.step()
                self.writer.add_scalar('Train Loss', loss.item(), epoch * len(self.train_loader) + i)

            with torch.no_grad():
                for i, (state, action) in enumerate(self.eval_loader):
                    output = self.forward(state)
                    loss = self.loss_function(output, action)
                    self.writer.add_scalar('Eval Loss', loss.item(), epoch * len(self.eval_loader) + i)
                    self.eval_losses.append(loss.item())
                    # Save the model if the validation loss is the best we've seen so far
                    if loss < best_loss:
                        best_loss = loss
                        self.save()
                        logger.debug(f"Model saved with loss {loss}, at epoch {epoch}")

                    # Interrupt training if the loss is not decreasing in the last 10 epochs
                    if len(self.eval_losses) > 10 and all(
                            self.eval_losses[-1] >= self.eval_losses[-i] for i in range(1, 11)):
                        logger.debug(f"Interrupting training at epoch {epoch} due to no decrease in loss")
                        break
        
        self.writer.close()

    def save(self, baseline_name='bc'):
        torch.save(self.state_dict(), baseline_path(baseline_name))


class AutomatumDataset(Dataset):
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


if __name__ == '__main__':
    policy_network = PolicyNetwork()
    policy_network.train(num_epochs=policy_network.EPOCHS, learning_rate=policy_network.LR)

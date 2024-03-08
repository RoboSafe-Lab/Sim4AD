"""
Behavioural cloning baseline for the automatum dataset
"""
import logging
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from shapely import Point
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from baselines.dataset import AutomatumDataset
from sim4ad.opendrive import Map, plot_map
from sim4ad.path_utils import baseline_path

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class LSTM(nn.Module):
    def __init__(self, name='bc'):
        super(LSTM, self).__init__()

        self.name = name

        expert_data = {"observations": [], "actions": []}
        for scenario in [
                        'hw-a9-appershofen-001-d8087340-8287-46b6-9612-869b09e68448',
                        'hw-a9-appershofen-002-2234a9ae-2de1-4ad4-9f43-65c2be9696d6',
                        'hw-a9-appershofen-003-6d6e3378-df9b-4130-8cbf-3437a77a309d',
                        'hw-a9-appershofen-004-e7ee10c6-a428-4416-bd84-cebb0476f565',
                        'hw-a9-appershofen-005-dc8c9357-291c-4b27-8c3d-98a048818efd',
                        'hw-a9-appershofen-006-7e386963-33f2-4e71-a750-76cc66791d43',
                        'hw-a9-appershofen-007-55244cc7-f80a-49dc-a29d-ed707b6ea4fb',
                        'hw-a9-appershofen-008-44cb097b-ce86-4d2d-b509-0e0c5b5b7ad5',
                        'hw-a9-appershofen-009-2caba3d6-ef31-48c8-b8e1-d9c6a300a68a',
                        'hw-a9-appershofen-011-b932653c-ea9c-424a-a8cc-51fb75ad9d59',
                        'hw-a9-appershofen-012-d696e4f3-70ac-45ac-9de1-79a2c9f6185c',
                        'hw-a9-appershofen-013-7e5d812c-a86f-468c-b9ed-d6888991eeb7',
                        ]:
            with open(f'scenarios/data/trainingdata/{scenario}/demonstration.pkl', 'rb') as f:
                new_expert_data = pickle.load(f)
                expert_data['observations'] += new_expert_data['observations']
                expert_data['actions'] += new_expert_data['actions']

        self.PADDING_VALUE = -1  # used to pad the LSTM input to the same length

        expert_states_all = pad_sequence(
                [torch.as_tensor(seq, dtype=torch.float32) for seq in expert_data['observations']],
                batch_first=True,
                padding_value=self.PADDING_VALUE)
        expert_actions_all = pad_sequence([torch.as_tensor(seq, dtype=torch.float32) for seq in expert_data['actions']],
                                          batch_first=True,
                                          padding_value=self.PADDING_VALUE)

        expert_states_train, expert_states_test, expert_actions_train, expert_actions_test = train_test_split(
            expert_states_all, expert_actions_all, test_size=0.2)

        # TODO: should we normalize the data?

        input_space = expert_states_train.shape[-1]
        action_space = expert_actions_train.shape[-1]

        self.BATCH_SIZE = 128  # Define your batch size # TODO: parameterize
        self.SHUFFLE = True  # shuffle your data
        self.EPOCHS = 10000  # Define the number of epochs # TODO: parameterize
        self.LR = 1e-3  # Define your learning rate # TODO: parameterize
        LSTM_HIDDEN_SIZE = 128
        FC_HIDDEN_SIZE = 512
        DROPOUT = 0.2  # TODO: remember to zero it out during evaluation!

        self.rnn = nn.LSTM(input_size=input_space,  hidden_size=LSTM_HIDDEN_SIZE, num_layers=2, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(LSTM_HIDDEN_SIZE, FC_HIDDEN_SIZE),
                                nn.ReLU(),
                                nn.Dropout(DROPOUT),
                                nn.Linear(FC_HIDDEN_SIZE, action_space),
                                nn.Dropout(DROPOUT),
                                nn.Tanh())  # TODO: check the range of acceleration/steering angle in the dataset
        self.float()

        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

        # TODO: should be 34 as we should not include x/y
        assert input_space == 34 and action_space == 2  # TODO: just for automatum dataset
        self.loss_function = nn.MSELoss(reduction="mean") # TODO nn.MSELoss(reduction="mean")  # TODO: is this the correct loss function / reduction?

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

    def compute_loss(self, trajectory, predicted_actions, actions):
        """ Only compute the loss for the time steps that were not padded """

        mask = ~(trajectory == self.PADDING_VALUE).all(dim=-1)
        loss = self.loss_function(predicted_actions[mask]*10, actions[mask]*10) # TODO: scaling?
        return loss

    def train(self, num_epochs=100, learning_rate=1e-3):
        best_loss = float('inf')

        self.writer = SummaryWriter(f'baselines/runs/bc')

        for epoch in tqdm(range(num_epochs)):
            # Training
            for i, (trajectory, actions) in enumerate(self.train_loader):
                # Divide the trajectory for each time step and feed the LSTM the history up until that point
                self.optimizer.zero_grad()
                predicted_actions = self.forward(trajectory)
                loss = self.compute_loss(trajectory, predicted_actions, actions)
                loss.backward()

                # Store the gradients
                ave_grads = []
                max_grads = []
                layers = []
                for name, param in self.named_parameters():
                    if param.requires_grad and ("bias" not in name):
                        layers.append(name)
                        ave_grads.append(param.grad.abs().mean().item())
                        max_grads.append(param.grad.abs().max().item())
                self.writer.add_scalar('Average Gradient', sum(ave_grads) / len(ave_grads), epoch * len(self.train_loader) + i)
                self.writer.add_scalar('Max Gradient', sum(max_grads) / len(max_grads), epoch * len(self.train_loader) + i)

                self.optimizer.step()
                self.writer.add_scalar('Train Loss', loss.item(), epoch * len(self.train_loader) + i)

            with torch.no_grad():
                for i, (trajectory, actions) in enumerate(self.eval_loader):
                    output = self.forward(trajectory)
                    loss = self.compute_loss(trajectory, output, actions)
                    self.writer.add_scalar('Eval Loss', loss.item(), epoch * len(self.eval_loader) + i)
                    self.eval_losses.append(loss.item())
                    logger.debug(f"Epoch {epoch}, batch {i}, eval loss {loss.item()}")
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

    def save(self):
        torch.save(self.state_dict(), baseline_path(self.name))


if __name__ == '__main__':
    policy_network = LSTM("bc-all-obs")
    policy_network.train(num_epochs=policy_network.EPOCHS, learning_rate=policy_network.LR)

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
from baselines.bc_model import LSTMModel
from sim4ad.data import DatasetDataLoader, ScenarioConfig
from sim4ad.opendrive import Map, plot_map
from sim4ad.path_utils import baseline_path, get_config_path

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class BCBaseline:
    def __init__(self, name: str, evaluation=False):

        self.name = name
        self.BATCH_SIZE = 128  # Define your batch size # TODO: parameterize
        self.SHUFFLE = True  # shuffle your data
        self.EPOCHS = 1000  # Define the number of epochs # TODO: parameterize
        self.LR = 1e-3  # Define your learning rate # TODO: parameterize
        LSTM_HIDDEN_SIZE = 128
        FC_HIDDEN_SIZE = 512
        DROPOUT = 0.2 if not evaluation else 0.0
        INPUT_SPACE = 34  # TODO: parameterize
        ACTION_SPACE = 2  # TODO: parameterize

        self.model = LSTMModel(INPUT_SPACE, ACTION_SPACE, LSTM_HIDDEN_SIZE, FC_HIDDEN_SIZE, DROPOUT)

        if evaluation:
            self.model.load_state_dict(torch.load(baseline_path(self.name)))
            self.model.eval()
        else:
            # We are training
            expert_data = {"observations": [], "actions": []}

            configs = ScenarioConfig.load(get_config_path("appershofen"))
            idx = configs.dataset_split["train"]
            episode_names = [x.recording_id for i, x in enumerate(configs.episodes) if i in idx]

            for episode in episode_names:
                with open(f'scenarios/data/trainingdata/{episode}/demonstration.pkl', 'rb') as f:
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

            assert expert_states_train.shape[-1] == INPUT_SPACE
            assert expert_actions_train.shape[-1] == ACTION_SPACE

            # Check cuda, cpu or mps
            if torch.cuda.is_available():
                self.device = torch.device('cuda')

                if torch.cuda.device_count() > 1:
                    self.model = nn.DataParallel(self.model)

            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')

            self.model.to(self.device)
            logger.info(f"The model is on device = {self.device}.")

            self.loss_function = nn.MSELoss(reduction="mean")
            self.loss_function.to(self.device)

            self.train_loader = DataLoader(AutomatumDataset(expert_states_train, expert_actions_train),
                                           batch_size=self.BATCH_SIZE, shuffle=self.SHUFFLE)
            self.eval_loader = DataLoader(AutomatumDataset(expert_states_test, expert_actions_test),
                                          batch_size=self.BATCH_SIZE, shuffle=self.SHUFFLE)

            self.eval_losses = []

            self.optimizer = optim.Adam(self.model.parameters(), lr=self.LR)

    def compute_loss(self, trajectory, predicted_actions, actions):
        """ Only compute the loss for the time steps that were not padded """

        mask = ~(trajectory == self.PADDING_VALUE).all(dim=-1)
        loss = self.loss_function((predicted_actions[mask]*10).to(self.device), (actions[mask]*10).to(self.device)) # TODO: scaling?
        return loss

    def train(self, num_epochs=100, learning_rate=1e-3):
        best_loss = float('inf')

        self.writer = SummaryWriter(f'baselines/runs/bc')

        for epoch in tqdm(range(num_epochs)):
            # Training
            for i, (trajectory, actions) in enumerate(self.train_loader):
                # Divide the trajectory for each time step and feed the LSTM the history up until that point
                self.optimizer.zero_grad()

                actions = actions.to(self.device)
                trajectory = trajectory.to(self.device)
                predicted_actions = self.model(trajectory)
                loss = self.compute_loss(trajectory, predicted_actions, actions)
                loss.backward()

                # Store the gradients
                ave_grads = []
                max_grads = []
                layers = []
                for name, param in self.model.named_parameters():
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
                    trajectory = trajectory.to(self.device)
                    actions = actions.to(self.device)
                    output = self.model(trajectory)
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
        torch.save(self.model.state_dict(), baseline_path(self.name))

    def __call__(self, trajectory):
        return self.model(trajectory)


if __name__ == '__main__':
    policy_network = BCBaseline("bc-all-obs-1.5_pi")
    policy_network.train(num_epochs=policy_network.EPOCHS, learning_rate=policy_network.LR)

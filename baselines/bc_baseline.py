"""
Behavioural cloning baseline for the automatum dataset
"""
import logging
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from shapely import Point
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.append(r"D:\IRLcode\Sim4AD")
from baselines.dataset import AutomatumDataset
from baselines.bc_model import LSTMModel
from sim4ad.common_constants import DEFAULT_SCENARIO, DEFAULT_CLUSTER, LSTM_PADDING_VALUE
from sim4ad.path_utils import baseline_path, get_config_path, get_processed_demonstrations
import argparse
from loguru import logger


class BCBaseline:
    def __init__(self, name: str, evaluation=False, cluster=DEFAULT_CLUSTER, scenario=DEFAULT_SCENARIO):

        logger.debug(f"Creating BC baseline with name {name}, evaluation = {evaluation}, cluster = {cluster}, "
                     f"scenario = {scenario}. If eval is true, the cluster is only used for the input and action space.")

        if evaluation:
            self.name = name
        else:
            self.name = f"{name}_cluster_{cluster}"
        self.BATCH_SIZE = 128
        self.SHUFFLE = True
        self.EPOCHS = 1000
        self.LR = 1e-3
        LSTM_HIDDEN_SIZE = 128
        FC_HIDDEN_SIZE = 512
        DROPOUT = 0.2 if not evaluation else 0.0

        self.INPUT_SPACE, self.ACTION_SPACE = self.load_datasets(evaluation=evaluation, cluster=cluster,
                                                                 scenario=scenario, get_dimensions_only=True)

        self.model = LSTMModel(self.INPUT_SPACE, self.ACTION_SPACE, LSTM_HIDDEN_SIZE, FC_HIDDEN_SIZE, DROPOUT)
        self.PADDING_VALUE = LSTM_PADDING_VALUE  # used to pad the LSTM input to the same length

        if evaluation:

            try:
                self.model.load_state_dict(torch.load(baseline_path(self.name), map_location=torch.device('cpu')))
                self.model.eval()
            except RuntimeError:
                # from https://stackoverflow.com/questions/44230907/keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict
                state_dict = torch.load(baseline_path(self.name), map_location=torch.device('cpu'))
                # create new OrderedDict that does not contain `module.`
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                # load params
                self.model.load_state_dict(new_state_dict)
            self.model.eval()
        else:
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

            expert_states_train, expert_actions_train, expert_states_valid, expert_actions_valid = self.load_datasets(
                evaluation=evaluation, cluster=cluster, scenario=scenario)

            self.train_loader = DataLoader(AutomatumDataset(expert_states_train, expert_actions_train),
                                           batch_size=self.BATCH_SIZE, shuffle=self.SHUFFLE)
            self.eval_loader = DataLoader(AutomatumDataset(expert_states_valid, expert_actions_valid),
                                          batch_size=self.BATCH_SIZE, shuffle=self.SHUFFLE)

            self.eval_losses = []

            self.optimizer = optim.Adam(self.model.parameters(), lr=self.LR)

    def compute_loss(self, trajectory, predicted_actions, actions):
        """ Only compute the loss for the time steps that were not padded """

        mask = ~(trajectory == self.PADDING_VALUE).all(dim=-1)
        loss = self.loss_function((predicted_actions[mask]*10).to(self.device), (actions[mask]*10).to(self.device))
        return loss

    def load_datasets(self, evaluation: bool, cluster: str, scenario: str, get_dimensions_only=False):
        def load_one_dataset(split_type):
            with open(get_processed_demonstrations(split_type=split_type, scenario=scenario, cluster=cluster), 'rb') as f:
                expert_data = pickle.load(f)
                if cluster == "All":
                    expert_data = expert_data["All"]
                else:
                    expert_data = expert_data["clustered"]

            if get_dimensions_only:
                # return observation_dimension, action_dimension
                return expert_data[0].observations[0].shape[-1], expert_data[0].actions[0].shape[-1]

            expert_states = pad_sequence(
                [torch.as_tensor(seq, dtype=torch.float32) for traj in expert_data for seq in traj.observations],
                batch_first=True,
                padding_value=self.PADDING_VALUE)
            expert_actions = pad_sequence([torch.as_tensor(seq, dtype=torch.float32) for traj in expert_data for seq in traj.actions],
                                              batch_first=True,
                                              padding_value=self.PADDING_VALUE)
            return expert_states, expert_actions

        if get_dimensions_only:
            return load_one_dataset(split_type="train")

        assert not evaluation, ("`evaluation` should be true only if the network is used to get prediction with already"
                                "trained weights.")
        expert_states_train, expert_actions_train = load_one_dataset(split_type="train")
        expert_states_valid, expert_actions_valid = load_one_dataset(split_type="valid")

        assert expert_states_train.shape[-1] == self.INPUT_SPACE
        assert expert_actions_train.shape[-1] == self.ACTION_SPACE

        return expert_states_train, expert_actions_train, expert_states_valid, expert_actions_valid

    def train(self, num_epochs=100, learning_rate=1e-3):
        best_loss = float('inf')

        self.writer = SummaryWriter(f'baselines/runs/bc')
        early_stopping = False

        for epoch in tqdm(range(num_epochs)):
            if early_stopping:
                break

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
                eval_losses_epoch = []
                for i, (trajectory, actions) in enumerate(self.eval_loader):
                    trajectory = trajectory.to(self.device)
                    actions = actions.to(self.device)
                    output = self.model(trajectory)
                    loss = self.compute_loss(trajectory, output, actions)
                    eval_losses_epoch.append(loss.item())

                epoch_eval_loss = np.mean(eval_losses_epoch)
                logger.debug(f"Epoch {epoch} eval loss {epoch_eval_loss}")
                self.writer.add_scalar('Eval Loss', epoch_eval_loss, epoch)
                self.eval_losses.append(epoch_eval_loss)
                # Save the model if the validation loss is the best we've seen so far
                if epoch_eval_loss < best_loss:
                    best_loss = epoch_eval_loss
                    self.save()
                    logger.debug(f"Model saved with loss {epoch_eval_loss}, at epoch {epoch}")
                        
                # Interrupt training if the loss is not decreasing in the last 'stop_after' epochs
                stop_after = 100  # epochs
                if epoch > 500 and all(
                        self.eval_losses[-1] >= self.eval_losses[-i] for i in range(1, stop_after)):
                    logger.debug(f"Interrupting training at epoch {epoch} due to no decrease in loss in past {stop_after} epochs")
                    early_stopping = True
        
        self.writer.close()

    def save(self):
        if isinstance(self.model, torch.nn.DataParallel):
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()

        torch.save(model_state_dict, baseline_path(self.name))

    def __call__(self, trajectory):
        return self.model(trajectory)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--cluster", type=str, default=DEFAULT_CLUSTER)
    args = args.parse_args()

    policy_network = BCBaseline("bc-all-obs-5_pi", cluster=args.cluster)
    policy_network.train(num_epochs=policy_network.EPOCHS, learning_rate=policy_network.LR)

import numpy as np
import torch
import torch.nn as nn


class LSTMModel(nn.Module):

    def __init__(self, input_space, action_space, lstm_hidden_size, fc_hidden_size, dropout):
        super(LSTMModel, self).__init__()

        self.rnn = nn.LSTM(input_size=input_space,  hidden_size=lstm_hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(lstm_hidden_size, fc_hidden_size),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Linear(fc_hidden_size, action_space),
                                nn.Dropout(dropout),
                                nn.Tanh())
        self.float()

        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, history: torch.Tensor):
        # out contains the hidden state for each time step of the trajectory, after all the layers
        if isinstance(history, list):
            history = torch.tensor(history, dtype=torch.float32)
        out, _ = self.rnn(history)
        out = self.fc(out)

        MAX_ACCELERATION = 1.5  # TODO: set these globally
        MAX_STEERING_ANGLE = np.pi

        # We want the output to be between -max_acceleration and max_acceleration and
        # between -max_steering_angle and max_steering_angle
        scaling = torch.tensor([MAX_ACCELERATION, MAX_STEERING_ANGLE]).to(out.device)
        return out * scaling

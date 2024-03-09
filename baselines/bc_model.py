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
                                nn.Tanh())  # TODO: check the range of acceleration/steering angle in the dataset
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

        return self.fc(out)
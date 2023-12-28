import torch
from torch import nn, device


class RnnLSTMModel(nn.Module):

    def __init__(self, input_size, hidden_size, rnn_units, output_size):
        super(RnnLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_units = rnn_units

        # Embedding layer
        self.embedding = nn.Embedding(input_size, hidden_size)

        # LSTM layer
        self.lstm = nn.LSTM(hidden_size, rnn_units, batch_first=True)

        # Dense output layer
        self.fc = nn.Linear(rnn_units, output_size)

    def forward(self, in_data, hidden=None, cell=None):
        # Embedding layer
        x = self.embedding(in_data)

        if hidden is None and cell is None:
            # Initialize hidden and cell states
            h0 = torch.zeros(1, x.size(0), self.rnn_units).to(x.device)
            c0 = torch.zeros(1, x.size(0), self.rnn_units).to(x.device)
        else:
            h0 = hidden
            c0 = cell

        # LSTM layers
        out, (h1, c1) = self.lstm(x, (h0, c0))

        # Dense output layer
        out = self.fc(out)
        return out, (h1, c1)

    def save_model(self, param):
        torch.save(self.state_dict(), param)


import torch
import torch.nn as nn
from torch import save, tensor, cat, load
import pathlib

class MyLSTM(nn.Module):
    def __init__(self, input_size=512, n_hidden=51):
        super(MyLSTM, self).__init__()
        self.n_hidden = n_hidden
        self.lstm = nn.LSTM(input_size, n_hidden)
        self.linear = nn.Linear(self.n_hidden, 1)

    def forward(self, x, future=0):
        outputs = []
        n_samples = x.size(0)

        h_t = torch.zeros(2, n_samples, self.n_hidden, dtype = torch.float32)
        c_t = torch.zeros(2, n_samples, self.n_hidden, dtype = torch.float32)
        h_t2 = torch.zeros(2, n_samples, self.n_hidden, dtype = torch.float32)
        c_t2 = torch.zeros(2, n_samples, self.n_hidden, dtype = torch.float32)
        
        for input_t in x.split(1, dim=0):
            h_t, c_t = self.lstm(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm(input_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)

        for i in range(future):
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm1(input_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)

        outputs = torch.cat(outputs, dim=1)
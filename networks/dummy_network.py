from math import prod
import torch.nn as nn

class DummyNet(nn.Module):
    def __init__(self, in_channels=5, out_channels=1, size=(128,16)):
        super().__init__()
        self.fc = nn.Linear(in_channels*prod(size), out_channels*prod(size))
        self.flatten = nn.Flatten()
        self.unflatten = nn.Unflatten(dim=1,unflattened_size=[out_channels,*size])   

    def forward(self, x):
        return nn.Sequential(self.flatten, self.fc, self.unflatten)(x)
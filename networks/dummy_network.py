from math import prod
import torch.nn as nn

class DummyNet(nn.Module):
    def __init__(self, in_channels=5, out_channels=1, size=(256,16)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels*prod(size), out_channels*prod(size)),
            nn.ReLU(),
            nn.Linear(out_channels*prod(size), out_channels*prod(size)),
            # nn.ReLU(),
            # nn.Linear(out_channels*prod(size), out_channels*prod(size)),
            nn.Unflatten(dim=1,unflattened_size=[out_channels,*size])
        )
        
    def forward(self, x):
        return self.net(x)

class DummyCNN(nn.Module):
    def __init__(self, in_channels=5, out_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, out_channels, kernel_size=3, stride=1, padding=1),
        )
        
    def forward(self, x):
        return self.net(x)
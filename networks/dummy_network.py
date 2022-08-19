import torch.nn as nn

class DummyNet(nn.Module):
    def __init__(self, in_channels=5, out_channels=1):
        super().__init__()
        self.fc = nn.Linear(in_channels*128*16, out_channels*128*16)
        self.flatten = nn.Flatten()
        self.unflatten = nn.Unflatten(dim=1,unflattened_size=[out_channels,128,16])        

    def forward(self, x):
        return nn.Sequential(self.flatten, self.fc, self.unflatten)(x)
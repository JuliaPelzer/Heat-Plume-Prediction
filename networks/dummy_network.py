import torch.nn as nn

class DummyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5*128*16, 1*128*16)
        self.flatten = nn.Flatten()
        self.unflatten = nn.Unflatten(dim=1,unflattened_size=[1,128,16])        

    def forward(self, x):
        return nn.Sequential(self.flatten, self.fc, self.unflatten)(x)
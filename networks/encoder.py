import torch.nn as nn
from torch import save, tensor, cat, load
import pathlib


class Encoder(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, init_features=32, depth=3, kernel_size=5):
        super().__init__()
        features = init_features
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        for _ in range(depth):
            self.encoders.append(Encoder._block(in_channels, features, kernel_size=kernel_size))
            self.pools.append(nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)))
            in_channels = features
            features *= 2
        self.encoders.append(Encoder._block(in_channels, features, kernel_size=kernel_size))

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=(9,1))

    def forward(self, x: tensor) -> tensor:
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            x = pool(x)
        x = self.encoders[-1](x)

        return self.conv(x)

    @staticmethod
    def _block(in_channels, features, kernel_size=5):
        return nn.Sequential(
            PaddingCircular(kernel_size),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=kernel_size,
                bias=True,
            ),
            nn.ReLU(inplace=True),      
            PaddingCircular(kernel_size),
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=kernel_size,
                bias=True,
            ),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),      
            PaddingCircular(kernel_size),
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=kernel_size,
                bias=True,
            ),        
            nn.ReLU(inplace=True),
        )
    
    def load(self, model_path:pathlib.Path, device:str = "cpu", model_name: str = "model.pt"):
        self.load_state_dict(load(model_path/model_name))
        self.to(device)

    def save(self, path:pathlib.Path, model_name: str = "model.pt"):
        save(self.state_dict(), path/model_name)

        model_structure = []
        for name, param in self.named_parameters():
            model_structure.append([name, param.shape])
        with open(path/"model_structure.txt", "w") as f:
            f.write(str(model_structure))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.zero_()

class PaddingCircular(nn.Module):
    def __init__(self, kernel_size, direction="horizontal"):
        super().__init__()
        self.pad_len = kernel_size//2
        self.direction = direction

    def forward(self, x:tensor) -> tensor:
        padding_vector = (self.pad_len,)*2 + (0,)*2
        result = nn.functional.pad(x, padding_vector, mode='circular')    
        return result
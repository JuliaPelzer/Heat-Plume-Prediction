import torch.nn as nn
from torch import save, tensor, cat, load
import pathlib

from processing.networks.unet import UNet
from processing.diff_conv2d.layers import *

class UNetHalfPad(UNet):
    def __init__(self, in_channels=2, out_channels=1, init_features=32, depth=3, kernel_size=5):
        super().__init__()
        features = init_features
        direction = "horizontal"
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        for _ in range(depth):
            self.encoders.append(UNetHalfPad._block(in_channels, features, kernel_size=kernel_size, direction=direction))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = features
            features *= 2
        self.encoders.append(UNetHalfPad._block(in_channels, features, kernel_size=kernel_size, direction=direction))

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for _ in range(depth):
            self.upconvs.append(nn.ConvTranspose2d(features, features//2, kernel_size=2, stride=2))
            self.decoders.append(UNetHalfPad._block(features, features//2, kernel_size=kernel_size, direction=direction))
            features = features // 2

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x: tensor) -> tensor:
        encodings = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            encodings.append(x)
            x = pool(x)
        x = self.encoders[-1](x)

        for upconv, decoder, encoding in zip(self.upconvs, self.decoders, reversed(encodings)):
            x = upconv(x)
            x = cat((x, encoding), dim=1)
            x = decoder(x)

        return self.conv(x)

    @staticmethod
    def _block(in_channels, features, kernel_size=5, direction="both"):
        return nn.Sequential(
            PaddingCircular(kernel_size, direction),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=kernel_size,
                bias=True,
            ),
            nn.ReLU(inplace=True),      
            PaddingCircular(kernel_size, direction),
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=kernel_size,
                bias=True,
            ),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),      
            PaddingCircular(kernel_size, direction),
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=kernel_size,
                bias=True,
            ),        
            nn.ReLU(inplace=True),
        )
    
class PaddingCircular(nn.Module):
    def __init__(self, kernel_size, direction="both"):
        super().__init__()
        self.pad_len = kernel_size//2
        self.direction = direction

    def forward(self, x:tensor) -> tensor:
        if self.direction == "both":
            padding_vector = (self.pad_len,)*4
            result = nn.functional.pad(x, padding_vector, mode='circular')    

        elif self.direction == "horizontal":
            padding_vector = (self.pad_len,)*2 + (0,)*2
            result = nn.functional.pad(x, padding_vector, mode='circular')    
            padding_vector = (0,)*2 + (self.pad_len,)*2  
            result = nn.functional.pad(result, padding_vector, mode='constant')     # or 'reflect'?

        elif self.direction == "vertical":
            padding_vector = (0,)*2 + (self.pad_len,)*2
            result = nn.functional.pad(x, padding_vector, mode='circular')    
            padding_vector = (self.pad_len,)*2 + (0,)*2
            result = nn.functional.pad(result, padding_vector, mode='constant')

        elif self.direction == "none":
            padding_vector = (self.pad_len,)*4
            result = nn.functional.pad(x, padding_vector, mode='constant')    
        return result
    
class UNetHalfPad2(UNet):
    def __init__(self, in_channels=2, out_channels=1, init_features=32, depth=3, kernel_size=5):
        super().__init__()
        features = init_features
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        for _ in range(depth):
            self.encoders.append(self._block(in_channels, features, kernel_size=kernel_size))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = features
            features *= 2
        self.encoders.append(self._block(in_channels, features, kernel_size=kernel_size))

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for _ in range(depth):
            self.upconvs.append(nn.ConvTranspose2d(features, features//2, kernel_size=2, stride=2))
            self.decoders.append(self._block(features, features//2, kernel_size=kernel_size))
            features = features // 2

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x: tensor) -> tensor:
        encodings = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            encodings.append(x)
            x = pool(x)
        x = self.encoders[-1](x)

        for upconv, decoder, encoding in zip(self.upconvs, self.decoders, reversed(encodings)):
            x = upconv(x)
            required_size = x.shape[2]
            start_pos = (encoding.shape[2] - required_size)//2
            encoding = encoding[:, :, start_pos:start_pos+required_size, :]
            x = cat((x, encoding), dim=1)
            x = decoder(x)

        return self.conv(x)

    @staticmethod
    def _block(in_channels, features, kernel_size=5):
        direction = "horizontal"
        return nn.Sequential(
            OneSidePadding(kernel_size, direction),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=kernel_size,
                bias=True,
            ),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
        )
    
class OneSidePadding(nn.Module):
    def __init__(self, kernel_size, direction):
        super().__init__()
        self.pad_len = kernel_size//2
        self.direction = direction

    def forward(self, x:tensor) -> tensor:
        if self.direction == "vertical":
            padding_vector = (0,)*2 + (self.pad_len,)*2  
            result = nn.functional.pad(x, padding_vector, mode='constant')     # or 'reflect'?
        elif self.direction == "horizontal":
            padding_vector = (self.pad_len,)*2 + (0,)*2
            result = nn.functional.pad(x, padding_vector, mode='constant')
        return result

    
class UNetBC(UNet):
    def __init__(self, in_channels=2, out_channels=1, init_features=32, depth=3, kernel_size=5):
        super().__init__(in_channels, out_channels, init_features, depth, kernel_size)

        features = init_features
        for _ in range(depth): features *= 2
        for _ in range(depth): features = features // 2
        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    @staticmethod
    def _block(in_channels, features, kernel_size=5, padding_mode="zeros"):
        return nn.Sequential(
            ExplicitConv2dLayer(
                in_channels, features, kernel_size, bias=True,),
            nn.ReLU(inplace=True),    
            ExplicitConv2dLayer(
                features, features, kernel_size, bias=True,),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),  
            ExplicitConv2dLayer(
                features, features, kernel_size, bias=True,),
            nn.ReLU(inplace=True),
        )
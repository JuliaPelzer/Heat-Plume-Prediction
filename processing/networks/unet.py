import torch.nn as nn
from torch import cat, tensor

from processing.networks.model import Model


class UNet(Model):
    def __init__(self, in_channels=2, out_channels=1, init_features=32, depth=3, kernel_size=5):
        super().__init__()
        self.features = init_features
        self.depth = depth
        self.kernel_size = kernel_size
        padding_mode =  "circular"            
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        for _ in range(depth):
            self.encoders.append(UNet._block(in_channels, self.features, kernel_size=kernel_size, padding_mode=padding_mode))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = self.features
            self.features *= 2
        self.encoders.append(UNet._block(in_channels, self.features, kernel_size=kernel_size, padding_mode=padding_mode))

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for _ in range(depth):
            self.upconvs.append(nn.ConvTranspose2d(self.features, self.features//2, kernel_size=2, stride=2))
            self.decoders.append(UNet._block(self.features, self.features//2, kernel_size=kernel_size, padding_mode=padding_mode))
            self.features = self.features // 2

        self.conv = nn.Conv2d(in_channels=self.features, out_channels=out_channels, kernel_size=1)

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
    def _block(in_channels, features, kernel_size=5, padding_mode="zeros"):
        return nn.Sequential(
            # PaddingCircular(kernel_size, direction="both"),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=kernel_size,
                padding="same",
                # padding_mode=padding_mode,
                bias=True,
            ),
            nn.ReLU(inplace=True),      
            # PaddingCircular(kernel_size, direction="both"),
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=kernel_size,
                padding="same",
                # padding_mode=padding_mode,
                bias=True,
            ),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),      
            # PaddingCircular(kernel_size, direction="both"),
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=kernel_size,
                padding="same",
                # padding_mode=padding_mode,
                bias=True,
            ),        
            nn.ReLU(inplace=True),
        )
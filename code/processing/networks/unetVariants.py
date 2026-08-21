import torch.nn as nn
from torch import cat, tensor

from processing.networks.model import Model

class UNet(Model):
    def __init__(self, in_channels:int=2, out_channels:int=1, init_features:int=32, depth:int=3, kernel_size:int=5, stride:int=1, dilation:int=1, activation:str="relu", norm:str="batchnorm", repeat_inner:bool=False, last_activation:bool=True):
        super().__init__()
        features = init_features
        activation = get_activation_fct(activation)

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        for _ in range(depth):
            self.encoders.append(self._block(in_channels, features, kernel_size=kernel_size, stride=stride, dilation=dilation, activation=activation, norm=norm, repeat_inner=repeat_inner))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = features
            features *= 2
        self.encoders.append(self._block(in_channels, features, kernel_size=kernel_size, stride=stride, dilation=dilation, activation=activation, norm=norm, repeat_inner=repeat_inner, last_activation=last_activation))

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for _ in range(depth):
            self.upconvs.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(features, features // 2, kernel_size=3, padding=1))
            )
            # self.upconvs.append(nn.ConvTranspose2d(self.features, self.features//2, kernel_size=2, stride=2))
            self.decoders.append(self._block(features, features//2, kernel_size=kernel_size, dilation=dilation, activation=activation, norm=norm, repeat_inner=repeat_inner, last_activation=last_activation))
            features = features // 2

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x: tensor) -> tensor:
        encodings = []
        for encoder, pool in zip(self.encoders[:-1], self.pools, strict=True): # TODO check train with zip(self.encoders[:-1], self.pools)
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
    def _block(in_channels, features, kernel_size=5, stride=1, dilation=1, activation=nn.ReLU(inplace=True), norm:str=None, repeat_inner=False, last_activation:bool=True):
        if repeat_inner:
            block = nn.Sequential(
                UNet._build_conv2d(in_channels, features, kernel_size, stride, dilation),
                UNet._build_norm2d(features, norm),
                activation,
                UNet._build_conv2d(features, features, kernel_size, stride, dilation),
            )
        else:
            block = nn.Sequential(
                UNet._build_conv2d(in_channels, features, kernel_size, stride, dilation),
                UNet._build_norm2d(features, norm),
            )

        if last_activation:
            block.append(activation)
        return block

    @staticmethod
    def _build_conv2d(in_channels, features, kernel_size, stride, dilation):
        return nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=features,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding="same",
                    bias=True)

    @staticmethod
    def _build_norm2d(features, norm):
        if norm.lower() == "batchnorm":
            return nn.BatchNorm2d(num_features=features)
        elif norm.lower() == "groupnorm":
            return nn.GroupNorm(num_groups=4, num_channels=features)
        else:
            return nn.Identity()


class UNetNoPad2(UNet):
    def __init__(self, in_channels:int=2, out_channels:int=1, init_features:int=32, depth:int=3, kernel_size:int=5, stride:int=1, dilation:int=1, activation:str="relu", norm:str="batchnorm", repeat_inner:bool=False):
        super().__init__()
        features = init_features
        activation = get_activation_fct(activation)

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        for _ in range(depth):
            self.encoders.append(self._block(in_channels, features, kernel_size=kernel_size, stride=stride, dilation=dilation, activation=activation, norm=norm, repeat_inner=repeat_inner))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = features
            features *= 2
        self.encoders.append(self._block(in_channels, features, kernel_size=kernel_size, stride=stride, dilation=dilation, activation=activation, norm=norm, repeat_inner=repeat_inner))

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for _ in range(depth):
            # self.upconvs.append(nn.ConvTranspose2d(features, features//2, kernel_size=2, stride=2))
            self.upconvs.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(features, features // 2, kernel_size=3, padding=1))
            )
            print("TODO in UNet we use Upsample and Conv2d to avoid checkerboard artifacts - check if it works!!")
            self.decoders.append(self._block(features, features//2, kernel_size=kernel_size, dilation=dilation, activation=activation, norm=norm, repeat_inner=repeat_inner))
            features = features // 2

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x: tensor) -> tensor:
        encodings = []
        for encoder, pool in zip(self.encoders[:-1], self.pools, strict=True): # TODO check train with zip(self.encoders[:-1], self.pools)
            x = encoder(x)
            encodings.append(x)
            x = pool(x)
        x = self.encoders[-1](x)

        for upconv, decoder, encoding in zip(self.upconvs, self.decoders, reversed(encodings)):
            x = upconv(x)
            required_size = x.shape[2:]
            start_pos = ((encoding.shape[2] - required_size[0])//2, (encoding.shape[3] - required_size[1])//2)
            encoding = encoding[:, :, start_pos[0]:start_pos[0]+required_size[0], start_pos[1]:start_pos[1]+required_size[1]]
            x = cat((x, encoding), dim=1)
            x = decoder(x)

        return self.conv(x)

    @staticmethod
    def _block(in_channels, features, kernel_size=5, stride=1, dilation=1, activation=nn.ReLU(inplace=True), norm:str=None, repeat_inner=False):
        if repeat_inner:
            return nn.Sequential(
                UNetNoPad2._build_conv2d(in_channels, features, kernel_size, stride, dilation),
                UNetNoPad2._build_norm2d(features, norm),
                activation,
                UNetNoPad2._build_conv2d(features, features, kernel_size, stride, dilation),
                activation,
            )
        else:
            return nn.Sequential(
                UNetNoPad2._build_conv2d(in_channels, features, kernel_size, stride, dilation),
                UNetNoPad2._build_norm2d(features, norm),
                activation,
            )


    @staticmethod
    def _build_conv2d(in_channels, features, kernel_size, stride, dilation):
        return nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=features,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding="valid", # "valid":= no padding, "same":=padding (default is with 0)
                    bias=True,
                )
  

def get_activation_fct(name:str):
    if name.lower() == "relu":
        return nn.ReLU(inplace=True)
    elif name.lower() == "leakyrelu":
        return nn.LeakyReLU(inplace=True)
    elif name.lower() == "sigmoid":
        return nn.Sigmoid()
    elif name.lower() == "tanh":
        return nn.Tanh()
      
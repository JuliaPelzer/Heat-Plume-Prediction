from code.processing.networks.model import Model
from code.utils import logging as log  # noqa: F401

import torch.nn as nn
import torch.nn.functional as F
from torch import cat, tensor


class UpsampleConv(nn.Module):
    """
    Replaces ConvTranspose2d to eliminate checkerboard artifacts.
    Primary Source: Odena et al., 2016 ("Deconvolution and Checkerboard Artifacts")
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same")

    @property
    def weight(self):
        return self.conv.weight

    @property
    def bias(self):
        return self.conv.bias

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.conv(x)


def kaiming_init(m):
    """
    Primary Source: He et al., 2015 ("Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification")
    """
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
        if m.weight is not None:
            nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class UNet(Model):
    def __init__(
        self, in_channels: int, out_channels: int, init_features: int, depth: int, kernel_size: int, norm: str, **kwargs
    ):
        super().__init__()
        self.features = init_features
        self.depth = depth
        self.kernel_size = kernel_size
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        for _ in range(depth):
            self.encoders.append(self._block(in_channels, self.features, kernel_size, norm))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = self.features
            self.features *= 2
        self.encoders.append(self._block(in_channels, self.features, kernel_size=kernel_size, norm=norm))

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for _ in range(depth):
            self.upconvs.append(UpsampleConv(self.features, self.features // 2))
            self.decoders.append(self._block(self.features, self.features // 2, kernel_size=kernel_size, norm=norm))
            self.features //= 2

        self.conv = nn.Conv2d(in_channels=self.features, out_channels=out_channels, kernel_size=1)
        self.apply(kaiming_init)

    def forward(self, x: tensor) -> tensor:
        encodings = []
        for encoder, pool in zip(self.encoders[:-1], self.pools, strict=True):
            x = encoder(x)
            encodings.append(x)
            x = pool(x)
        x = self.encoders[-1](x)

        for upconv, decoder, encoding in zip(self.upconvs, self.decoders, reversed(encodings), strict=True):
            x = upconv(x)
            if x.shape[2:] != encoding.shape[2:]:
                diffY = encoding.size(2) - x.size(2)
                diffX = encoding.size(3) - x.size(3)
                x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            x = cat((x, encoding), dim=1)
            x = decoder(x)

        return self.conv(x)

    @staticmethod
    def _block(in_channels, features, kernel_size, norm):
        use_bias = norm is None or not norm
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=features, kernel_size=kernel_size, padding="same", bias=use_bias
            ),
            UNet._build_norm2d(features, norm),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=features, out_channels=features, kernel_size=kernel_size, padding="same", bias=use_bias
            ),
            UNet._build_norm2d(features, norm),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _build_norm2d(features, norm):
        if not norm:
            return nn.Identity()
        norm_type = norm.lower()
        if norm_type == "batchnorm":
            return nn.BatchNorm2d(num_features=features)
        elif norm_type == "groupnorm":
            return nn.GroupNorm(num_groups=4, num_channels=features)
        elif norm_type == "instancenorm":
            return nn.InstanceNorm2d(num_features=features, affine=True)
        elif norm_type == "identity":
            return nn.Identity()
        raise ValueError(f"Normalization type '{norm}' not recognized.")


def get_activation_fct(name: str):
    name = name.lower()
    if name == "relu":
        return nn.ReLU
    elif name == "leakyrelu":
        return nn.LeakyReLU
    elif name == "sigmoid":
        return nn.Sigmoid
    elif name == "tanh":
        return nn.Tanh
    raise ValueError(f"Activation function '{name}' not recognized.")


class UNetNoPad2(UNet):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        init_features: int,
        depth: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        activation: str,
        norm: str,
        repeat_inner: bool = False,
    ):
        super(UNet, self).__init__()

        features = init_features
        act_cls = get_activation_fct(activation)
        self.stride = stride

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        for _ in range(depth):
            self.encoders.append(
                self._block(
                    in_channels,
                    features,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    activation=act_cls,
                    norm=norm,
                    repeat_inner=repeat_inner,
                )
            )
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = features
            features *= 2
        self.encoders.append(
            self._block(
                in_channels,
                features,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                activation=act_cls,
                norm=norm,
                repeat_inner=repeat_inner,
            )
        )

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for _ in range(depth):
            self.upconvs.append(UpsampleConv(features, features // 2))
            self.decoders.append(
                self._block(
                    features,
                    features // 2,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    activation=act_cls,
                    norm=norm,
                    repeat_inner=repeat_inner,
                    stride=stride,
                )
            )
            features //= 2

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)
        self.apply(kaiming_init)

    def forward(self, x: tensor) -> tensor:
        encodings = []
        for encoder, pool in zip(self.encoders[:-1], self.pools, strict=True):
            x = encoder(x)
            encodings.append(x)
            x = pool(x)
        x = self.encoders[-1](x)

        for upconv, decoder, encoding in zip(self.upconvs, self.decoders, reversed(encodings), strict=True):
            x = upconv(x)
            if x.shape[2:] != encoding.shape[2:]:
                diffY = encoding.size(2) - x.size(2)
                diffX = encoding.size(3) - x.size(3)
                encoding = encoding[
                    :,
                    :,
                    diffY // 2 : encoding.size(2) - (diffY - diffY // 2),
                    diffX // 2 : encoding.size(3) - (diffX - diffX // 2),
                ]
            x = cat((x, encoding), dim=1)
            x = decoder(x)

        return self.conv(x)

    @staticmethod
    def _block(in_channels, features, kernel_size, stride, dilation, activation, norm: str, repeat_inner):
        use_bias = norm is None or not norm
        layers = [
            UNetNoPad2._build_conv2d(in_channels, features, kernel_size, stride, dilation, use_bias),
            UNet._build_norm2d(features, norm),
            activation(inplace=True) if activation in (nn.ReLU, nn.LeakyReLU) else activation(),
        ]

        if repeat_inner:
            layers.extend(
                [
                    UNetNoPad2._build_conv2d(features, features, kernel_size, stride, dilation, use_bias),
                    UNet._build_norm2d(features, norm),
                    activation(inplace=True) if activation in (nn.ReLU, nn.LeakyReLU) else activation(),
                ]
            )
        return nn.Sequential(*layers)

    @staticmethod
    def _build_conv2d(in_channels, features, kernel_size, stride, dilation, bias):
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=features,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding="valid",
            bias=bias,
        )

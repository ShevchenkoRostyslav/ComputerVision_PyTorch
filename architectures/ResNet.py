from typing import List

import torch
from torch import nn
import torch.nn.functional as F
from functools import partial

def activation_fn(name):
    return nn.ModuleDict([
    ['relu', nn.ReLU(inplace=True)],
    ['leaky_relu', nn.LeakyReLU(inplace=True)],
    ['selu', nn.SELU(inplace=True)],
    ['none', nn.Identity()]
])[name]

class Conv2dSamePad(nn.Conv2d):
    """Implementation of the 2D convolution layer with same padding.

    """
    def __init__(self, *args, **kwargs):
        super(Conv2dSamePad, self).__init__(*args, **kwargs)
        # dynamic same padding based on the kernel_size
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)


Conv3x3SamePad = partial(Conv2dSamePad, kernel_size=3, bias=False)


def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(conv(in_channels, out_channels, *args, **kwargs), nn.BatchNorm2d(out_channels))


class ResidualBlock(nn.Module):
    """Base residual block that takes an input, applies some NN blocks on it and sum it up
    with the original input. An activation function is applied ofterwards.
    """
    def __init__(self, in_channels: int, out_channels: int, activation: str = 'relu'):
        super(ResidualBlock, self).__init__()
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation_fn(activation)
        self.blocks = nn.Identity()
        self.shortcut = nn.Identity()

    def forward(self, x):
        x_in = x
        if self.apply_shortcut: x_in = self.shortcut(x)
        x_out = self.blocks(x)
        x_out += x_in
        x_out = self.activation(x_out)
        return x_out

    @property
    def apply_shortcut(self):
        return self.in_channels != self.out_channels


class ResNetResidualBlock(ResidualBlock):
    """ResNet Implementation of the ResidualBlock.
    blocks:
    Residual block is comprised of two Conv2d 3x3 layers with BatchNormalization in between.
    1st Conv2D block has #mid_channels and the last one #out_channels
    shortcuts:
    If the in_channels = out_channels the shortcut uses the Identity mapping,
    otherwise 1x1 Convolution is performed to match the dimensionality.
    """
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int, activation: str = 'relu', downsampling_stride: int = 1):
        super(ResNetResidualBlock, self).__init__(in_channels, out_channels, activation)
        self.shortcut = conv_bn(in_channels, out_channels, nn.Conv2d, kernel_size=1, stride=downsampling_stride, bias=False)
        self.blocks = nn.Sequential(
            conv_bn(in_channels, mid_channels, Conv3x3SamePad, bias=False, stride=downsampling_stride),
            activation_fn(activation),
            conv_bn(mid_channels, out_channels, Conv3x3SamePad, bias=False, stride=1),
        )


class ResNetResidualLayer(nn.Module):
    """A single layer of ResNet model.
    It consists of number of ResidualBlocks stacked one after the other.

    """
    def __init__(self, in_channels: int, out_channels: int, block=ResNetResidualBlock, n=1, *args, **kwargs):
        super(ResNetResidualLayer, self).__init__()
        downsampling_stride = 2 if in_channels != out_channels else 1
        self.blocks = nn.Sequential(
            block(in_channels, out_channels, out_channels, downsampling_stride=downsampling_stride, *args, **kwargs),
            *[block(out_channels, out_channels, out_channels, downsampling_stride=1, *args, **kwargs) for _ in range(n-1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x


class ResNetGate(nn.Module):
    """The first Conv layer + Max Pooling Layer, before the ResidualBlocks.

    """
    def __init__(self, in_channels: int = 3, out_channels: int = 64, activation: str = 'relu', *args, **kwargs):
        super(ResNetGate, self).__init__()
        self.gate = nn.Sequential(
            conv_bn(in_channels, out_channels, conv=nn.Conv2d, kernel_size=7, stride=2, padding=3,
                    bias=False, *args, **kwargs),
            activation_fn(activation),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        x = self.gate(x)
        return x


class ResNetFeatureExtractor(nn.Module):
    """

    """
    def __init__(self, in_channels: int = 3, filter_sizes: List[int] = [64, 128, 256, 512],
                 layer_sizes: List[int] = [2, 2, 2, 2], activation: str = 'relu', *args, **kwargs):
        super(ResNetFeatureExtractor, self).__init__()
        self.gate = ResNetGate(in_channels, out_channels=filter_sizes[0], activation=activation)
        self.blocks = nn.ModuleList([
            ResNetResidualLayer(in_channels=filter_sizes[0], out_channels=filter_sizes[0], activation=activation,
                                n=layer_sizes[0], *args, **kwargs),
            *[ResNetResidualLayer(in_channels=in_ch, out_channels=out_ch, activation=activation, n=layer_size, *args, **kwargs)
              for in_ch, out_ch, layer_size in zip(filter_sizes[:], filter_sizes[1:], layer_sizes[1:])]
        ])

    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x


class ResNetClassifier(nn.Module):
    """

    """
    def __init__(self, in_channels: int, n_classes: int):
        super(ResNetClassifier, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.decoder = nn.Linear(in_channels, n_classes)

    def forward(self, x):
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.decoder(x)
        return x


class ResNet(nn.Module):
    """Implementation of the ResNet for CIFAR-10 from:
    "Deep Residual Learning for Image Recognition", K. He et al.
    https://arxiv.org/pdf/1512.03385.pdf
    """

    def __init__(self, in_channels: int = 3, n_classes: int = 10, *args, **kwargs):
        super().__init__()
        self.feature_extractor = ResNetFeatureExtractor(in_channels=in_channels, *args, **kwargs)
        self.classifier = ResNetClassifier(in_channels=self.feature_extractor.blocks[-1].blocks[-1].out_channels,
                                           n_classes=n_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs

def resnet18(in_channels, n_classes, *args, **kwargs):
    return ResNet(in_channels, n_classes, layer_sizes=[2, 2, 2, 2], *args, **kwargs)

def resnet34(in_channels, n_classes, *args, **kwargs):
    return ResNet(in_channels, n_classes, layer_sizes=[3, 4, 6, 3], *args, **kwargs)

def resnet50(in_channels, n_classes, *args, **kwargs):
    return ResNet(in_channels, n_classes, layer_sizes=[3, 4, 6, 3], *args, **kwargs)

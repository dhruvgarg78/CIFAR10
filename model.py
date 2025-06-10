# model.py

"""
Defines ConvBlock, DepthwiseSeparableConv, and CIFAR10CustomNet model architecture.
"""

import torch
import torch.nn as nn

# Flexible ConvBlock used throughout the model
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1, dilation=1, groups=1,
                 use_bn=True, use_relu=True, use_dropout=True, dropout_p=0.0):
        super(ConvBlock, self).__init__()

        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=False))

        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))

        if use_relu:
            layers.append(nn.ReLU())

        if use_dropout:
            layers.append(nn.Dropout(dropout_p))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

# Depthwise Separable Convolution block
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels,
                 use_bn=True, use_relu=True, use_dropout=True, dropout_p=0.0):
        super(DepthwiseSeparableConv, self).__init__()

        layers = []
        # Depthwise convolution
        layers.append(nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False))
        # Pointwise convolution
        layers.append(nn.Conv2d(in_channels, out_channels, 1, bias=False))

        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))

        if use_relu:
            layers.append(nn.ReLU())

        if use_dropout:
            layers.append(nn.Dropout(dropout_p))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

# CIFAR-10 Custom CNN model
class CIFAR10CustomNet(nn.Module):
    def __init__(self, num_classes=10, dropout=0.055):
        super(CIFAR10CustomNet, self).__init__()

        # C1 block: 2 ConvBlocks
        self.c1 = nn.Sequential(
            ConvBlock(3, 32, use_bn=True, use_relu=True, use_dropout=True, dropout_p=dropout),
            ConvBlock(32, 64, use_bn=True, use_relu=True, use_dropout=True, dropout_p=dropout),
        )

        # C2 block: alternating dilated and depthwise convs
        self.c2 = nn.Sequential(
            ConvBlock(64, 64, padding=2, dilation=2, use_bn=True, use_relu=True, use_dropout=True, dropout_p=dropout),
            DepthwiseSeparableConv(64, 64, use_bn=True, use_relu=True, use_dropout=True, dropout_p=dropout),
            ConvBlock(64, 64, padding=2, dilation=2, use_bn=True, use_relu=True, use_dropout=True, dropout_p=dropout),
            DepthwiseSeparableConv(64, 64, use_bn=True, use_relu=True, use_dropout=True, dropout_p=dropout)
        )

        # C3 block: 4 Depthwise Separable convs
        self.dwconv = nn.Sequential(
            *[DepthwiseSeparableConv(64, 64, use_bn=True, use_relu=True, use_dropout=True, dropout_p=dropout) for _ in range(4)]
        )

        # C40 block: convs with stride and dilation changes
        self.c40 = nn.Sequential(
            ConvBlock(64, 32, use_bn=True, use_relu=True, use_dropout=True, dropout_p=dropout),
            ConvBlock(32, 32, stride=2, use_bn=True, use_relu=True, use_dropout=True, dropout_p=dropout),
            ConvBlock(32, 32, dilation=2, use_bn=True, use_relu=True, use_dropout=False),
            ConvBlock(32, 64, dilation=2, use_bn=True, use_relu=True, use_dropout=False),
            DepthwiseSeparableConv(64, 64, use_bn=True, use_relu=True, use_dropout=False)
        )

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 1x1 ConvBlock to reduce channels
        self.c5 = nn.Sequential(
            ConvBlock(64, 64, kernel_size=1, padding=0, use_bn=False, use_relu=False, use_dropout=False)
        )

        # Final FC layer
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.dwconv(x)
        x = self.c40(x)
        x = self.gap(x)
        x = self.c5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

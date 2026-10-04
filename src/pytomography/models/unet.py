# Copyright (c) 2026 H. Wei
# SPDX-License-Identifier: MIT
# See the LICENSE file in the repository root for license terms.

"""Residual 3D feature U-Net for patch-trained deep kernels."""
from numbers import Integral
import torch
from torch import nn
from torch.nn import functional as F


def _conv(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1),
        nn.BatchNorm3d(out_channels), nn.LeakyReLU(.01, inplace=False),
    )


def _double_conv(in_channels, out_channels):
    return nn.Sequential(_conv(in_channels, out_channels), _conv(out_channels, out_channels))


class UNet3D(nn.Module):
    """Three-level 3D U-Net with additive skips and nonnegative features.

    Args:
        in_channels (int): Number of input channels, normally 1 for a prior.
        base_channels (int): Encoder width; successive levels use 1, 2, 4,
            and 8 times this width.
        feature_channels (int): Number of learned features per voxel.

    Inputs and outputs follow [batch, channels, X, Y, Z] order. Spatial sizes
    must be at least 16; odd sizes are supported. The architecture adapts the
    user's deep_kem.py with fixed-scale interpolation and high-side cropping
    to maintain coordinates during aligned tiled inference.
    """
    def __init__(self, in_channels=1, base_channels=16, feature_channels=16):
        super().__init__()
        for name, value in [('in_channels', in_channels), ('base_channels', base_channels),
                            ('feature_channels', feature_channels)]:
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        self.in_channels = in_channels
        self.feature_channels = feature_channels
        channels = [base_channels * 2**level for level in range(4)]
        self.encoders = nn.ModuleList([_double_conv(in_channels, channels[0])] +
            [_double_conv(channels[i-1], channels[i]) for i in range(1, 4)])
        self.down = nn.ModuleList([_conv(c, c, stride=2) for c in channels[:3]])
        self.up = nn.ModuleList([
            nn.Sequential(nn.Conv3d(channels[i+1], channels[i], 1),
                          nn.BatchNorm3d(channels[i]), nn.LeakyReLU(.01))
            for i in range(3)
        ])
        self.decoders = nn.ModuleList([_double_conv(c, c) for c in channels[:3]])
        self.output = nn.Sequential(nn.Conv3d(channels[0], feature_channels, 3, padding=1),
                                    nn.BatchNorm3d(feature_channels), nn.ReLU())
        for layer in self.modules():
            if isinstance(layer, nn.Conv3d):
                nn.init.kaiming_normal_(layer.weight, a=.01, nonlinearity='leaky_relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, value):
        if value.ndim != 5 or value.shape[1] != self.in_channels or min(value.shape[2:]) < 16:
            raise ValueError('Expected [B, in_channels, X, Y, Z] with spatial sizes >= 16')
        skips = []
        for level in range(4):
            value = self.encoders[level](value)
            if level < 3:
                skips.append(value)
                value = self.down[level](value)
        for level in (2, 1, 0):
            value = self.up[level](F.interpolate(value, scale_factor=2,
                mode='trilinear', align_corners=False))
            shape = skips[level].shape[2:]
            value = value[:, :, :shape[0], :shape[1], :shape[2]]
            value = self.decoders[level](value + skips[level])
        return self.output(value)

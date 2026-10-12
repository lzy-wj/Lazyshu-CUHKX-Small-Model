#!/usr/bin/env python3
import torch
from torch import nn


class DepthwiseResidual3D(nn.Module):
    def __init__(self, inputs, outputs, stride=(1, 1, 1)):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(inputs, inputs, 3, stride=stride, padding=1, groups=inputs, bias=False),
            nn.GroupNorm(min(8, inputs), inputs),
            nn.SiLU(),
            nn.Conv3d(inputs, outputs, 1, bias=False),
            nn.GroupNorm(min(8, outputs), outputs),
        )
        self.residual = (
            nn.Identity()
            if inputs == outputs and stride == (1, 1, 1)
            else nn.Sequential(
                nn.Conv3d(inputs, outputs, 1, stride=stride, bias=False),
                nn.GroupNorm(min(8, outputs), outputs),
            )
        )
        self.activation = nn.SiLU()

    def forward(self, values):
        return self.activation(self.layers(values) + self.residual(values))


class PoseHeatmapCNN(nn.Module):
    def __init__(self, channels=15, classes=40):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(channels, 48, (3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2), bias=False),
            nn.GroupNorm(8, 48),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            DepthwiseResidual3D(48, 48),
            DepthwiseResidual3D(48, 64, (1, 2, 2)),
            DepthwiseResidual3D(64, 64),
            DepthwiseResidual3D(64, 96, (2, 2, 2)),
            DepthwiseResidual3D(96, 96),
            DepthwiseResidual3D(96, 160, (2, 2, 2)),
            DepthwiseResidual3D(160, 160),
            DepthwiseResidual3D(160, 256, (2, 1, 1)),
            DepthwiseResidual3D(256, 256),
        )
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, classes))

    def forward(self, values):
        values = self.blocks(self.stem(values))
        features = torch.cat([values.mean(dim=(2, 3, 4)), values.amax(dim=(2, 3, 4))], dim=1)
        return self.head(features)

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

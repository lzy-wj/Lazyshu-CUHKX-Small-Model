import torch
from torch import nn
from torch.nn import functional


class ResidualBlock(nn.Module):
    def __init__(self, inputs, outputs, stride=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(inputs, outputs, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
            nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
            nn.BatchNorm2d(outputs),
        )
        self.skip = (
            nn.Sequential(
                nn.Conv2d(inputs, outputs, 1, stride, bias=False), nn.BatchNorm2d(outputs)
            )
            if inputs != outputs or stride != 1
            else nn.Identity()
        )

    def forward(self, values):
        return functional.relu(self.layers(values) + self.skip(values), inplace=True)


class ThermalGRUNet(nn.Module):
    def __init__(self, classes=40, feature_dimension=256, hidden_dimension=256, dropout=0.4):
        super().__init__()
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 7, 2, 3, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            ResidualBlock(32, 64), ResidualBlock(64, 64),
            ResidualBlock(64, 128, 2), ResidualBlock(128, 128),
            ResidualBlock(128, feature_dimension, 2), ResidualBlock(feature_dimension, feature_dimension),
            nn.AdaptiveAvgPool2d(1),
        )
        self.recurrent = nn.GRU(
            feature_dimension,
            hidden_dimension,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dimension * 2, 64), nn.Tanh(), nn.Linear(64, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dimension * 2, 256), nn.BatchNorm1d(256), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(256, classes),
        )

    def forward(self, video):
        batch, steps, channels, height, width = video.shape
        features = self.frame_encoder(video.reshape(batch * steps, channels, height, width))
        features = features.flatten(1).reshape(batch, steps, -1)
        features, _ = self.recurrent(features)
        weights = functional.softmax(self.attention(features), dim=1)
        return self.classifier((features * weights).sum(dim=1))

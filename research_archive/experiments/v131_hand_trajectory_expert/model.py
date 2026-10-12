import torch
from torch import nn


JOINTS = (8, 9, 10, 11, 12, 13, 14, 15, 16)
PARENTS = (0, 8, 9, 8, 11, 12, 8, 14, 15)


class ResidualTemporalBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(width, width, 5, padding=2 * dilation, dilation=dilation, bias=False),
            nn.GroupNorm(8, width),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(width, width, 1, bias=False),
            nn.GroupNorm(8, width),
        )

    def forward(self, values):
        return torch.nn.functional.gelu(values + self.layers(values))


class HandTrajectoryExpert(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.input = nn.Sequential(
            nn.Conv1d(81, width, 1, bias=False),
            nn.GroupNorm(8, width),
            nn.GELU(),
        )
        self.convolution = nn.Sequential(
            ResidualTemporalBlock(width, 1),
            ResidualTemporalBlock(width, 2),
            ResidualTemporalBlock(width, 4),
            ResidualTemporalBlock(width, 8),
        )
        self.recurrent = nn.GRU(
            width, width, batch_first=True, bidirectional=True
        )
        layer = nn.TransformerEncoderLayer(
            width * 2,
            nhead=8,
            dim_feedforward=width * 4,
            dropout=0.15,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=1, norm=nn.LayerNorm(width * 2)
        )
        self.attention = nn.Linear(width * 2, 1)
        self.head = nn.Sequential(
            nn.Linear(width * 6, width * 2),
            nn.LayerNorm(width * 2),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(width * 2, 40),
        )

    @staticmethod
    def features(values):
        pose = values[:, :, :51].reshape(*values.shape[:2], 17, 3)
        centered = pose - pose[:, :, :1]
        scale = torch.linalg.vector_norm(centered[:, :, 8], dim=-1).median(1).values
        scale = scale.clamp(0.15, 1.2)[:, None, None, None]
        centered = (centered / scale).clamp(-6, 6)
        selected = centered[:, :, JOINTS]
        parent = centered[:, :, PARENTS]
        bones = selected - parent
        velocity = torch.diff(selected, dim=1, prepend=selected[:, :1]) * 10
        return torch.cat([selected, bones, velocity], dim=-1).flatten(2)

    def forward(self, values):
        features = self.features(values).transpose(1, 2)
        features = self.convolution(self.input(features)).transpose(1, 2)
        features, _ = self.recurrent(features)
        features = self.transformer(features)
        weights = self.attention(features).softmax(dim=1)
        attended = (features * weights).sum(dim=1)
        pooled = torch.cat(
            [
                attended,
                features.mean(dim=1),
                features.amax(dim=1),
            ],
            dim=1,
        )
        return self.head(pooled)


import math

import torch
from torch import nn
from torch.nn import functional


def position_encoding(length, dimensions, device, dtype):
    positions = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    frequencies = torch.exp(torch.arange(0, dimensions, 2, device=device).float() * (-math.log(10000.0) / dimensions))
    phases = positions * frequencies
    return torch.stack((phases.sin(), phases.cos()), dim=-1).flatten(1).to(dtype)


class SpatialVideoTokens(nn.Module):
    def __init__(self, kind, feedforward=384):
        super().__init__()
        from src.scratch_models import ResidualSpatial

        self.kind = kind
        if kind == "patch_tokens":
            self.frames = nn.Conv2d(3, 192, kernel_size=16, stride=16)
        else:
            self.frames = nn.Sequential(ResidualSpatial(3, 32), ResidualSpatial(32, 64),
                                        ResidualSpatial(64, 128), ResidualSpatial(128, 192))
        self.spatial = nn.TransformerEncoderLayer(192, 4, feedforward, dropout=0.1, batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoderLayer(192, 4, feedforward, dropout=0.1, batch_first=True, norm_first=True)
        self.normalization = nn.LayerNorm(192)

    def forward(self, values):
        batch, steps, channels, height, width = values.shape
        if height % 16 or width % 16 or steps != 16:
            raise ValueError("Spatial study requires 16 frames and image sides divisible by 16")
        features = self.frames(values.reshape(batch * steps, channels, height, width))
        if self.kind == "cnn_pooled":
            features = functional.adaptive_avg_pool2d(features, 2)
        rows, columns = features.shape[-2:]
        row_position = position_encoding(rows, 96, features.device, features.dtype)[:, None].expand(-1, columns, -1)
        column_position = position_encoding(columns, 96, features.device, features.dtype)[None].expand(rows, -1, -1)
        spatial_position = torch.cat((row_position, column_position), dim=-1).reshape(rows * columns, 192)
        features = features.flatten(2).transpose(1, 2) + spatial_position
        features = self.spatial(features)
        features = features.reshape(batch, steps, rows * columns, 192).transpose(1, 2)
        features = features.reshape(batch * rows * columns, steps, 192)
        features = self.temporal(features + position_encoding(steps, 192, features.device, features.dtype))
        features = features.reshape(batch, rows * columns, steps, 192).transpose(1, 2)
        return self.normalization(features.reshape(batch, steps * rows * columns, 192))

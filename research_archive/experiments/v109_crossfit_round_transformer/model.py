#!/usr/bin/env python3
import numpy as np
import torch
from torch import nn


def normalized_log_probabilities(probabilities):
    values = np.asarray(probabilities, dtype=np.float32)
    values = np.clip(values, 1e-7, None)
    values /= values.sum(axis=1, keepdims=True)
    return np.log(values)


class CrossFittedRoundTransformer(nn.Module):
    def __init__(self, dimension=128, maximum_length=16, residual_scale=0.5):
        super().__init__()
        self.maximum_length = maximum_length
        self.residual_scale = residual_scale
        self.input_projection = nn.Sequential(
            nn.Linear(80, dimension), nn.LayerNorm(dimension), nn.GELU()
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, maximum_length, dimension)
        )
        self.domain_embedding = nn.Embedding(4, dimension)
        self.length_embedding = nn.Embedding(maximum_length + 1, dimension)
        layer = nn.TransformerEncoderLayer(
            dimension,
            nhead=8,
            dim_feedforward=dimension * 3,
            dropout=0.2,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=2, norm=nn.LayerNorm(dimension), enable_nested_tensor=False
        )
        self.residual = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(dimension, 40),
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, features, anchors, padding, domains, lengths):
        tokens = self.input_projection(features)
        tokens = tokens + self.position_embedding[:, : tokens.shape[1]]
        tokens = tokens + self.domain_embedding(domains).unsqueeze(1)
        tokens = tokens + self.length_embedding(lengths).unsqueeze(1)
        encoded = self.encoder(tokens, src_key_padding_mask=padding)
        return anchors + self.residual_scale * self.residual(encoded)

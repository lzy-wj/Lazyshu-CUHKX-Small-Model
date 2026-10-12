#!/usr/bin/env python3
import torch
from torch import nn


class ObjectTokenTransformer(nn.Module):
    def __init__(self, dimension=192, phases=8, modalities=3):
        super().__init__()
        self.phases = phases
        self.modalities = modalities
        self.input_projection = nn.Sequential(
            nn.Linear(80 * 6, dimension), nn.LayerNorm(dimension), nn.GELU()
        )
        self.class_token = nn.Parameter(torch.zeros(1, 1, dimension))
        self.phase_embedding = nn.Parameter(torch.zeros(1, phases, 1, dimension))
        self.modality_embedding = nn.Parameter(
            torch.zeros(1, 1, modalities, dimension)
        )
        layer = nn.TransformerEncoderLayer(
            dimension,
            nhead=6,
            dim_feedforward=dimension * 3,
            dropout=0.2,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=3, norm=nn.LayerNorm(dimension), enable_nested_tensor=False
        )
        self.head = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(dimension, 40),
        )
        for parameter in (
            self.class_token,
            self.phase_embedding,
            self.modality_embedding,
        ):
            nn.init.trunc_normal_(parameter, std=0.02)

    def forward(self, features):
        batch_size = features.shape[0]
        tokens = self.input_projection(features.flatten(3))
        tokens = tokens + self.phase_embedding + self.modality_embedding
        tokens = tokens.flatten(1, 2)
        class_token = self.class_token.expand(batch_size, -1, -1)
        encoded = self.encoder(torch.cat([class_token, tokens], dim=1))
        return self.head(encoded[:, 0])

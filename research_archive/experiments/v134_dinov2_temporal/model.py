#!/usr/bin/env python3
import torch
from torch import nn

from src.scratch_models import ConvLSTMTokens, SkeletonTokens


class DINOv2TemporalMultimodal(nn.Module):
    def __init__(self, dimension=192, temporal_depth=2, fusion_depth=3, heads=6):
        super().__init__()
        self.visual_projection = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, dimension))
        self.visual_modality_embedding = nn.Parameter(torch.zeros(1, 3, 1, dimension))
        self.time_embedding = nn.Parameter(torch.zeros(1, 1, 8, dimension))
        temporal_layer = nn.TransformerEncoderLayer(
            dimension,
            heads,
            dimension * 4,
            dropout=0.15,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.visual_temporal = nn.TransformerEncoder(
            temporal_layer, temporal_depth, enable_nested_tensor=False
        )
        self.visual_attention = nn.Linear(dimension, 1)
        self.skeleton = SkeletonTokens("coordinates")
        self.imu = ConvLSTMTokens()
        self.skeleton_projection = nn.Linear(192, dimension)
        self.imu_projection = nn.Linear(192, dimension)
        self.class_token = nn.Parameter(torch.zeros(1, 1, dimension))
        self.modality_embedding = nn.Parameter(torch.zeros(1, 5, dimension))
        fusion_layer = nn.TransformerEncoderLayer(
            dimension,
            heads,
            dimension * 4,
            dropout=0.15,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(
            fusion_layer, fusion_depth, enable_nested_tensor=False
        )
        self.normalization = nn.LayerNorm(dimension)
        self.head = nn.Sequential(
            nn.Linear(dimension, dimension), nn.GELU(), nn.Dropout(0.25), nn.Linear(dimension, 40)
        )
        self.auxiliary_heads = nn.ModuleList([nn.Linear(dimension, 40) for _ in range(5)])
        nn.init.trunc_normal_(self.visual_modality_embedding, std=0.02)
        nn.init.trunc_normal_(self.time_embedding, std=0.02)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.modality_embedding, std=0.02)

    def forward(self, visual, skeleton, imu, visual_mask, sensor_mask):
        batch, modalities, steps, _ = visual.shape
        visual_tokens = self.visual_projection(visual)
        visual_tokens = visual_tokens + self.visual_modality_embedding + self.time_embedding[:, :, :steps]
        visual_tokens = self.visual_temporal(visual_tokens.reshape(batch * modalities, steps, -1))
        visual_weights = self.visual_attention(visual_tokens).softmax(dim=1)
        visual_pooled = (visual_tokens * visual_weights).sum(dim=1).reshape(batch, modalities, -1)
        skeleton_tokens = self.skeleton(skeleton)
        imu_tokens = self.imu(imu)
        skeleton_pooled = self.skeleton_projection(skeleton_tokens.mean(dim=1))
        imu_pooled = self.imu_projection(imu_tokens.mean(dim=1))
        modality_tokens = torch.cat(
            [visual_pooled, skeleton_pooled.unsqueeze(1), imu_pooled.unsqueeze(1)], dim=1
        )
        modality_tokens = modality_tokens + self.modality_embedding
        availability = torch.cat([visual_mask, sensor_mask[:, :2]], dim=1).bool()
        class_token = self.class_token.expand(batch, -1, -1)
        tokens = torch.cat([class_token, modality_tokens], dim=1)
        padding_mask = torch.cat(
            [torch.zeros(batch, 1, dtype=torch.bool, device=tokens.device), ~availability], dim=1
        )
        fused = self.fusion(tokens, src_key_padding_mask=padding_mask)
        embedding = self.normalization(fused[:, 0])
        auxiliary = torch.stack(
            [head(token) for head, token in zip(self.auxiliary_heads, modality_tokens.unbind(1))],
            dim=1,
        )
        return self.head(embedding), {"embedding": embedding, "auxiliary": auxiliary}

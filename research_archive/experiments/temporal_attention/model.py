import math

import torch
from torch import nn
from torch.nn import functional

from src.r2plus1d_models import VideoEncoder
from src.scratch_models import ConvLSTMTokens, SkeletonTokens


def normalized_time_coordinates(length, device):
    return (torch.arange(length, device=device, dtype=torch.float32) + 0.5) / length


def continuous_time_embedding(coordinates, dimension):
    half = dimension // 2
    frequencies = torch.logspace(0, math.log10(16), half, device=coordinates.device)
    angles = 2 * math.pi * coordinates[:, None] * frequencies[None]
    embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
    if embedding.shape[-1] < dimension:
        embedding = functional.pad(embedding, (0, dimension - embedding.shape[-1]))
    return embedding


class RelativeTimeAttention(nn.Module):
    def __init__(self, dimension=192, heads=6, modalities=3, dropout=0.15):
        super().__init__()
        if dimension % heads:
            raise ValueError("Attention dimension must be divisible by head count")
        self.dimension = dimension
        self.heads = heads
        self.head_dimension = dimension // heads
        self.modalities = modalities
        self.dropout = dropout
        self.qkv = nn.Linear(dimension, dimension * 3)
        self.output = nn.Linear(dimension, dimension)
        self.relative_offset = nn.Parameter(torch.zeros(heads, modalities, modalities))
        self.relative_log_scale = nn.Parameter(torch.full((heads, modalities, modalities), -2.0))
        self.modality_pair_bias = nn.Parameter(torch.zeros(heads, modalities, modalities))
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.zeros_(self.qkv.bias)

    def relative_bias(self, times, modality_ids):
        valid = modality_ids >= 0
        safe_ids = modality_ids.clamp_min(0)
        query_ids = safe_ids[:, None].expand(-1, len(modality_ids))
        key_ids = safe_ids[None, :].expand(len(modality_ids), -1)
        offsets = 0.5 * self.relative_offset.tanh()[:, query_ids, key_ids]
        scales = functional.softplus(self.relative_log_scale)[:, query_ids, key_ids]
        pair_bias = self.modality_pair_bias[:, query_ids, key_ids]
        delta = times[:, None] - times[None, :]
        bias = pair_bias - scales * (delta.unsqueeze(0) - offsets).abs()
        timed_pairs = valid[:, None] & valid[None, :]
        return bias * timed_pairs.unsqueeze(0)

    def forward(self, values, times, modality_ids, padding_mask=None):
        batch, length, _ = values.shape
        qkv = self.qkv(values).reshape(batch, length, 3, self.heads, self.head_dimension)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention_mask = self.relative_bias(times, modality_ids).unsqueeze(0)
        if padding_mask is not None:
            attention_mask = attention_mask.expand(batch, -1, -1, -1).clone()
            attention_mask.masked_fill_(padding_mask[:, None, None, :], torch.finfo(values.dtype).min)
        attended = functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, length, self.dimension)
        return self.output(attended)


class RelativeTimeEncoderLayer(nn.Module):
    def __init__(self, dimension=192, heads=6, feedforward=576, dropout=0.15, modalities=3):
        super().__init__()
        self.normalization1 = nn.LayerNorm(dimension)
        self.normalization2 = nn.LayerNorm(dimension)
        self.attention = RelativeTimeAttention(dimension, heads, modalities, dropout)
        self.linear1 = nn.Linear(dimension, feedforward)
        self.linear2 = nn.Linear(feedforward, dimension)
        self.dropout_attention = nn.Dropout(dropout)
        self.dropout_hidden = nn.Dropout(dropout)
        self.dropout_output = nn.Dropout(dropout)

    def forward(self, values, times, modality_ids, padding_mask):
        normalized = self.normalization1(values)
        values = values + self.dropout_attention(
            self.attention(normalized, times, modality_ids, padding_mask)
        )
        normalized = self.normalization2(values)
        hidden = self.dropout_hidden(functional.gelu(self.linear1(normalized)))
        return values + self.dropout_output(self.linear2(hidden))


class RelativeTimeFusion(nn.Module):
    def __init__(self, dimension=192, heads=6, feedforward=576, depth=2, dropout=0.15, modalities=3):
        super().__init__()
        self.dimension = dimension
        self.time_gain = nn.Parameter(torch.tensor(0.1))
        self.layers = nn.ModuleList([
            RelativeTimeEncoderLayer(dimension, heads, feedforward, dropout, modalities)
            for _ in range(depth)
        ])

    def forward(self, class_token, streams, padding_mask):
        device = class_token.device
        embeddings = []
        coordinates = []
        modality_ids = []
        for modality, stream in enumerate(streams):
            times = normalized_time_coordinates(stream.shape[1], device)
            temporal = continuous_time_embedding(times, self.dimension).to(stream.dtype)
            embeddings.append(stream + self.time_gain.to(stream.dtype) * temporal.unsqueeze(0))
            coordinates.append(times)
            modality_ids.append(torch.full((stream.shape[1],), modality, device=device, dtype=torch.long))
        values = torch.cat([class_token] + embeddings, dim=1)
        times = torch.cat([torch.tensor([0.5], device=device), *coordinates])
        ids = torch.cat([torch.full((1,), -1, device=device, dtype=torch.long), *modality_ids])
        for layer in self.layers:
            values = layer(values, times, ids, padding_mask)
        return values


class TemporalAttentionR2Plus1DClassifier(nn.Module):
    def __init__(self, weights=None, dropout=0.3, dimension=192, fusion_depth=2, input_channels=4):
        super().__init__()
        self.mode = "multimodal"
        self.video = VideoEncoder(input_channels, weights)
        self.video_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, 40))
        self.skeleton = SkeletonTokens()
        self.imu = ConvLSTMTokens()
        self.video_projection = nn.Linear(512, dimension)
        self.class_token = nn.Parameter(torch.zeros(1, 1, dimension))
        self.modality_embedding = nn.Parameter(torch.zeros(1, 3, 1, dimension))
        self.fusion = RelativeTimeFusion(
            dimension=dimension,
            heads=6,
            feedforward=dimension * 3,
            depth=fusion_depth,
            dropout=0.15,
            modalities=3,
        )
        self.normalization = nn.LayerNorm(dimension)
        self.head = nn.Sequential(
            nn.Linear(dimension, dimension), nn.GELU(), nn.Dropout(dropout), nn.Linear(dimension, 40)
        )
        self.skeleton_head = nn.Linear(dimension, 40)
        self.imu_head = nn.Linear(dimension, 40)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.modality_embedding, std=0.02)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, video, skeleton=None, imu=None, sensor_mask=None, visual_mask=None):
        if skeleton is None or imu is None or sensor_mask is None:
            raise ValueError("Multimodal mode requires skeleton, IMU, and masks")
        video_map = self.video(video)
        video_features = video_map.mean(dim=(2, 3, 4))
        video_logits = self.video_head(video_features)
        video_tokens = self.video_projection(video_map.mean(dim=(3, 4)).transpose(1, 2))
        skeleton_tokens = self.skeleton(skeleton)
        imu_tokens = self.imu(imu)
        streams = [video_tokens, skeleton_tokens, imu_tokens]
        streams = [tokens + self.modality_embedding[:, index] for index, tokens in enumerate(streams)]
        class_token = self.class_token.expand(video.shape[0], -1, -1)
        video_mask = torch.zeros(
            video.shape[0], video_tokens.shape[1], dtype=torch.bool, device=video.device
        )
        skeleton_mask = (~sensor_mask[:, 0].bool()).unsqueeze(1).expand(-1, skeleton_tokens.shape[1])
        imu_mask = (~sensor_mask[:, 1].bool()).unsqueeze(1).expand(-1, imu_tokens.shape[1])
        padding_mask = torch.cat(
            [torch.zeros(video.shape[0], 1, dtype=torch.bool, device=video.device),
             video_mask, skeleton_mask, imu_mask],
            dim=1,
        )
        fused = self.fusion(class_token, streams, padding_mask)
        output = video_logits + self.head(self.normalization(fused[:, 0]))
        auxiliary = {
            "video": video_logits,
            "skeleton": self.skeleton_head(skeleton_tokens.mean(dim=1)),
            "imu": self.imu_head(imu_tokens.mean(dim=1)),
        }
        return output, auxiliary

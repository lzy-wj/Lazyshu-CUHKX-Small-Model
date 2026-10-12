import torch
from torch import nn
from torch.nn import functional
from torchvision import models

from src.scratch_models import ConvLSTMTokens, SkeletonTokens


class FullSceneMultiModal(nn.Module):
    def __init__(self, pretrained=True, dimension=192, depth=2, dropout=0.25):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        backbone.fc = nn.Identity()
        self.visual = backbone
        self.visual_projection = nn.Linear(512, dimension)
        self.skeleton = SkeletonTokens("coordinates")
        self.imu = ConvLSTMTokens()
        self.class_token = nn.Parameter(torch.zeros(1, 1, dimension))
        self.modality_embedding = nn.Parameter(torch.zeros(1, 5, 1, dimension))
        self.temporal_embedding = nn.Parameter(torch.zeros(1, 16, dimension))
        layer = nn.TransformerEncoderLayer(
            dimension,
            nhead=6,
            dim_feedforward=dimension * 3,
            dropout=0.15,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(layer, depth, norm=nn.LayerNorm(dimension))
        self.head = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension, 40),
        )
        self.visual_head = nn.Linear(dimension, 40)
        self.skeleton_head = nn.Linear(dimension, 40)
        self.imu_head = nn.Linear(dimension, 40)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.modality_embedding, std=0.02)
        nn.init.trunc_normal_(self.temporal_embedding, std=0.02)

    def add_embeddings(self, tokens, modality):
        temporal = functional.interpolate(
            self.temporal_embedding.transpose(1, 2),
            size=tokens.shape[1],
            mode="linear",
            align_corners=True,
        ).transpose(1, 2)
        return tokens + temporal + self.modality_embedding[:, modality]

    def forward(self, images, skeleton, imu, visual_mask, sensor_mask):
        batch, modalities, frames = images.shape[:3]
        features = self.visual(images.flatten(0, 2)).reshape(
            batch, modalities, frames, 512
        )
        visual_tokens = self.visual_projection(features)
        streams = [
            self.add_embeddings(visual_tokens[:, modality], modality)
            for modality in range(modalities)
        ]
        skeleton_tokens = self.add_embeddings(self.skeleton(skeleton), 3)
        imu_tokens = self.add_embeddings(self.imu(imu), 4)
        streams.extend([skeleton_tokens, imu_tokens])
        tokens = torch.cat(
            [self.class_token.expand(batch, -1, -1), *streams], dim=1
        )
        masks = [
            (~visual_mask[:, modality].bool()).unsqueeze(1).expand(-1, frames)
            for modality in range(modalities)
        ]
        masks.extend(
            [
                (~sensor_mask[:, 0].bool()).unsqueeze(1).expand(
                    -1, skeleton_tokens.shape[1]
                ),
                (~sensor_mask[:, 1].bool()).unsqueeze(1).expand(
                    -1, imu_tokens.shape[1]
                ),
            ]
        )
        padding_mask = torch.cat(
            [
                torch.zeros(batch, 1, dtype=torch.bool, device=images.device),
                *masks,
            ],
            dim=1,
        )
        fused = self.fusion(tokens, src_key_padding_mask=padding_mask)
        visual_logits = self.visual_head(visual_tokens.mean(dim=2))
        visual_weights = visual_mask / visual_mask.sum(dim=1, keepdim=True).clamp_min(1)
        auxiliary = {
            "visual": visual_logits,
            "skeleton": self.skeleton_head(skeleton_tokens.mean(dim=1)),
            "imu": self.imu_head(imu_tokens.mean(dim=1)),
        }
        output = self.head(fused[:, 0]) + (
            visual_logits * visual_weights.unsqueeze(2)
        ).sum(dim=1)
        return output, auxiliary

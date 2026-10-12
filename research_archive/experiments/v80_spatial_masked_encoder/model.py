#!/usr/bin/env python3
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LOOP_ROOT = ROOT / "experiments/v24_iterative_loop"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LOOP_ROOT))

from iterative_model import IterativeR2Plus1D, build_base_model, checkpoint_state

class SpatialMaskedMultimodalEncoder(IterativeR2Plus1D):
    def __init__(
        self,
        base,
        dimension=512,
        depth=4,
        heads=8,
        feedforward_multiplier=3,
        residual_scale=0.25,
        reconstruction_mask_probability=0.35,
    ):
        if dimension % heads:
            raise ValueError("dimension must be divisible by heads")
        if not 0 <= reconstruction_mask_probability < 1:
            raise ValueError("reconstruction mask probability must be in [0, 1)")
        super().__init__(base, iterations=1)
        self.dimension = dimension
        self.depth = depth
        self.heads = heads
        self.feedforward_multiplier = feedforward_multiplier
        self.residual_scale = residual_scale
        self.reconstruction_mask_probability = reconstruction_mask_probability
        self.visual_projection = nn.Linear(256, dimension)
        self.skeleton_projection = nn.Linear(192, dimension)
        self.imu_projection = nn.Linear(192, dimension)
        self.class_projection = nn.Linear(192, dimension)
        self.visual_temporal_embedding = nn.Parameter(torch.zeros(1, 8, 1, dimension))
        self.visual_spatial_embedding = nn.Parameter(torch.zeros(1, 1, 4, dimension))
        self.sensor_temporal_embedding = nn.Parameter(torch.zeros(1, 16, dimension))
        self.modality_embedding = nn.Parameter(torch.zeros(1, 5, 1, dimension))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dimension))
        layer = nn.TransformerEncoderLayer(
            dimension,
            nhead=heads,
            dim_feedforward=dimension * feedforward_multiplier,
            dropout=0.15,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=depth,
            enable_nested_tensor=False,
        )
        self.output_normalization = nn.LayerNorm(dimension)
        self.residual_head = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(dimension, 40),
        )
        self.reconstruction_decoder = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.GELU(),
            nn.Linear(dimension, dimension),
        )
        for parameter in (
            self.visual_temporal_embedding,
            self.visual_spatial_embedding,
            self.sensor_temporal_embedding,
            self.modality_embedding,
            self.mask_token,
        ):
            nn.init.trunc_normal_(parameter, std=0.02)
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def encode_with_intermediate(
        self,
        video,
        skeleton,
        imu,
        radar,
        sensor_mask,
        visual_mask,
    ):
        captured_visual = []
        captured_skeleton = []
        captured_imu = []
        handles = [
            self.base.video.layer3.register_forward_hook(
                lambda module, inputs, output: captured_visual.append(output)
            ),
            self.base.skeleton.register_forward_hook(
                lambda module, inputs, output: captured_skeleton.append(output)
            ),
            self.base.imu.register_forward_hook(
                lambda module, inputs, output: captured_imu.append(output)
            ),
        ]
        try:
            encoded = self.encode_once(video, skeleton, imu, radar, sensor_mask, visual_mask)
        finally:
            for handle in handles:
                handle.remove()
        if (
            len(captured_visual) != 1
            or captured_visual[0].shape[0] != video.shape[0] * 3
            or len(captured_skeleton) != 1
            or len(captured_imu) != 1
        ):
            raise RuntimeError("Failed to capture single-pass modality encoder outputs")
        return (
            encoded,
            captured_visual[0].chunk(3, dim=0),
            captured_skeleton[0],
            captured_imu[0],
        )

    def base_prediction(self, encoded, batch_size, device):
        class_token = self.base.class_token.expand(batch_size, -1, -1)
        class_padding = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
        padding_mask = torch.cat([class_padding, encoded.padding_mask], dim=1)
        tokens = torch.cat([class_token, encoded.stream_tokens], dim=1)
        fused = self.base.fusion(tokens, src_key_padding_mask=padding_mask)
        fused_class = self.base.normalization(fused[:, :1])
        return fused_class, self.classify(fused_class[:, 0], encoded)

    def visual_tokens(self, intermediate_maps, visual_mask):
        streams = []
        targets = []
        masks = []
        position = self.visual_temporal_embedding + self.visual_spatial_embedding
        for modality, feature_map in enumerate(intermediate_maps):
            pooled = functional.adaptive_avg_pool3d(feature_map, (8, 2, 2))
            pooled = pooled.permute(0, 2, 3, 4, 1).reshape(feature_map.shape[0], 8, 4, 256)
            content = functional.layer_norm(
                self.visual_projection(pooled),
                (self.dimension,),
            )
            target = content.detach().reshape(feature_map.shape[0], 32, self.dimension)
            available = visual_mask[:, modality].bool().unsqueeze(1).expand(-1, 32)
            sampled = (
                torch.rand(available.shape, device=feature_map.device)
                < self.reconstruction_mask_probability
            ) & available if self.training and self.reconstruction_mask_probability > 0 else torch.zeros_like(available)
            masked_content = torch.where(
                sampled.reshape(feature_map.shape[0], 8, 4, 1),
                self.mask_token.reshape(1, 1, 1, self.dimension),
                content,
            )
            tokens = masked_content + position + self.modality_embedding[:, modality].unsqueeze(1)
            streams.append(tokens.reshape(feature_map.shape[0], 32, self.dimension))
            targets.append(target)
            masks.append(sampled)
        return torch.cat(streams, dim=1), torch.cat(targets, dim=1), torch.cat(masks, dim=1)

    def forward(
        self,
        video,
        skeleton=None,
        imu=None,
        radar=None,
        sensor_mask=None,
        visual_mask=None,
        return_training_details=False,
    ):
        if radar is not None or self.base.use_radar:
            raise ValueError("V80 is a frozen no-radar experiment")
        if visual_mask is None:
            visual_mask = torch.ones(video.shape[0], 3, device=video.device)
        encoded, intermediate_maps, skeleton_features, imu_features = self.encode_with_intermediate(
            video,
            skeleton,
            imu,
            radar,
            sensor_mask,
            visual_mask,
        )
        base_class, base_logits = self.base_prediction(encoded, video.shape[0], video.device)
        visual_tokens, reconstruction_targets, reconstruction_mask = self.visual_tokens(
            intermediate_maps,
            visual_mask,
        )
        skeleton_tokens = self.skeleton_projection(skeleton_features.detach())
        imu_tokens = self.imu_projection(imu_features.detach())
        skeleton_tokens = (
            skeleton_tokens
            + self.sensor_temporal_embedding
            + self.modality_embedding[:, 3]
        )
        imu_tokens = (
            imu_tokens
            + self.sensor_temporal_embedding
            + self.modality_embedding[:, 4]
        )
        class_token = self.class_projection(base_class.detach())
        tokens = torch.cat([class_token, visual_tokens, skeleton_tokens, imu_tokens], dim=1)
        visual_padding = torch.cat(
            [
                (~visual_mask[:, modality].bool()).unsqueeze(1).expand(-1, 32)
                for modality in range(3)
            ],
            dim=1,
        )
        padding_mask = torch.cat(
            [
                torch.zeros(video.shape[0], 1, dtype=torch.bool, device=video.device),
                visual_padding,
                (~sensor_mask[:, 0].bool()).unsqueeze(1).expand(-1, 16),
                (~sensor_mask[:, 1].bool()).unsqueeze(1).expand(-1, 16),
            ],
            dim=1,
        )
        fused = self.encoder(tokens, src_key_padding_mask=padding_mask)
        fused = self.output_normalization(fused)
        logits = base_logits.detach() + self.residual_scale * self.residual_head(fused[:, 0])
        details = {"base_logits": base_logits.detach()}
        if return_training_details:
            details.update(
                {
                    "reconstruction_predictions": self.reconstruction_decoder(fused[:, 1:97]),
                    "reconstruction_targets": reconstruction_targets,
                    "reconstruction_mask": reconstruction_mask,
                }
            )
        return logits, details

    def inference_parameter_count(self):
        return sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if not name.startswith("reconstruction_decoder.")
        )


def build_from_source(
    source_path,
    dimension=512,
    depth=4,
    heads=8,
    feedforward_multiplier=3,
    residual_scale=0.25,
    reconstruction_mask_probability=0.35,
):
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if config["mode"] != "shared_multimodal" or config.get("use_radar", False):
        raise ValueError("V80 requires a shared no-radar source")
    if config.get("visual_strategy") != "joint":
        raise ValueError("V80 currently requires joint visual encoding")
    base = build_base_model(config)
    base.load_state_dict(checkpoint_state(checkpoint), strict=True)
    model = SpatialMaskedMultimodalEncoder(
        base,
        dimension=dimension,
        depth=depth,
        heads=heads,
        feedforward_multiplier=feedforward_multiplier,
        residual_scale=residual_scale,
        reconstruction_mask_probability=reconstruction_mask_probability,
    )
    for parameter in model.base.parameters():
        parameter.requires_grad_(False)
    return model, config

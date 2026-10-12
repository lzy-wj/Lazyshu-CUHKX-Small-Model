#!/usr/bin/env python3
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.v24_iterative_loop.iterative_model import build_base_model, checkpoint_state


LABEL_GROUPS = (
    tuple(range(0, 6)) + tuple(range(37, 40)),
    tuple(range(6, 17)),
    tuple(range(17, 28)),
    tuple(range(28, 37)),
)
GROUP_NAMES = ("body_care", "food_cleaning", "desk_device", "motion_posture")


class PhaseAlignedLabelExperts(nn.Module):
    def __init__(
        self,
        base,
        dimension=384,
        phase_depth=1,
        expert_depth=2,
        heads=6,
        feedforward_multiplier=3,
        residual_scale=0.25,
        phases=16,
    ):
        super().__init__()
        if dimension % heads:
            raise ValueError("Expert dimension must be divisible by attention heads")
        if sorted(label for group in LABEL_GROUPS for label in group) != list(range(40)):
            raise ValueError("Label expert groups must partition all 40 classes")
        self.base = base
        self.dimension = dimension
        self.phases = phases
        self.residual_scale = residual_scale
        self.visual_projection = nn.Linear(256, dimension)
        self.skeleton_projection = nn.Linear(192, dimension)
        self.imu_projection = nn.Linear(192, dimension)
        self.modality_embedding = nn.Parameter(torch.zeros(1, 1, 5, dimension))
        self.phase_embedding = nn.Parameter(torch.zeros(1, phases, dimension))
        phase_layer = nn.TransformerEncoderLayer(
            dimension,
            nhead=heads,
            dim_feedforward=dimension * feedforward_multiplier,
            dropout=0.15,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.phase_encoder = nn.TransformerEncoder(
            phase_layer,
            num_layers=phase_depth,
            enable_nested_tensor=False,
        )
        self.expert_class_tokens = nn.Parameter(torch.zeros(len(LABEL_GROUPS), 1, dimension))
        self.expert_encoders = nn.ModuleList()
        self.expert_normalizations = nn.ModuleList()
        self.expert_heads = nn.ModuleList()
        for group in LABEL_GROUPS:
            layer = nn.TransformerEncoderLayer(
                dimension,
                nhead=heads,
                dim_feedforward=dimension * feedforward_multiplier,
                dropout=0.15,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.expert_encoders.append(
                nn.TransformerEncoder(layer, num_layers=expert_depth, enable_nested_tensor=False)
            )
            self.expert_normalizations.append(nn.LayerNorm(dimension))
            head = nn.Sequential(
                nn.Linear(dimension, dimension),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(dimension, len(group) + 1),
            )
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
            self.expert_heads.append(head)
        for parameter in (
            self.modality_embedding,
            self.phase_embedding,
            self.expert_class_tokens,
        ):
            nn.init.trunc_normal_(parameter, std=0.02)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def capture_base_features(self, video, skeleton, imu, sensor_mask, visual_mask):
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
            base_logits, _ = self.base(
                video,
                skeleton=skeleton,
                imu=imu,
                sensor_mask=sensor_mask,
                visual_mask=visual_mask,
            )
        finally:
            for handle in handles:
                handle.remove()
        if len(captured_visual) != 1 or captured_visual[0].shape[0] != video.shape[0] * 3:
            raise RuntimeError("Failed to capture joint visual layer3 features")
        if len(captured_skeleton) != 1 or len(captured_imu) != 1:
            raise RuntimeError("Failed to capture temporal sensor features")
        return (
            base_logits.detach(),
            captured_visual[0].chunk(3, dim=0),
            captured_skeleton[0],
            captured_imu[0],
        )

    def phase_tokens(self, visual_features, skeleton_features, imu_features, availability):
        visual_tokens = []
        for values in visual_features:
            values = values.mean(dim=(3, 4))
            values = functional.interpolate(
                values,
                size=self.phases,
                mode="linear",
                align_corners=True,
            ).transpose(1, 2)
            visual_tokens.append(self.visual_projection(values))
        skeleton = functional.interpolate(
            skeleton_features.transpose(1, 2),
            size=self.phases,
            mode="linear",
            align_corners=True,
        ).transpose(1, 2)
        imu = functional.interpolate(
            imu_features.transpose(1, 2),
            size=self.phases,
            mode="linear",
            align_corners=True,
        ).transpose(1, 2)
        streams = visual_tokens + [self.skeleton_projection(skeleton), self.imu_projection(imu)]
        tokens = torch.stack(streams, dim=2) + self.modality_embedding
        batch_size = tokens.shape[0]
        tokens = tokens.reshape(batch_size * self.phases, 5, self.dimension)
        phase_mask = (~availability.bool()).unsqueeze(1).expand(-1, self.phases, -1)
        phase_mask = phase_mask.reshape(batch_size * self.phases, 5)
        encoded = self.phase_encoder(tokens, src_key_padding_mask=phase_mask)
        weights = (~phase_mask).unsqueeze(2).to(encoded.dtype)
        pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        return pooled.reshape(batch_size, self.phases, self.dimension) + self.phase_embedding

    def forward(
        self,
        video,
        skeleton=None,
        imu=None,
        radar=None,
        sensor_mask=None,
        visual_mask=None,
    ):
        if radar is not None or self.base.use_radar:
            raise ValueError("V95 uses the frozen no-radar V23 base")
        if skeleton is None or imu is None or sensor_mask is None:
            raise ValueError("V95 requires Skeleton, IMU, and sensor masks")
        if visual_mask is None:
            visual_mask = torch.ones(video.shape[0], 3, device=video.device)
        base_logits, visual_features, skeleton_features, imu_features = self.capture_base_features(
            video,
            skeleton,
            imu,
            sensor_mask,
            visual_mask,
        )
        availability = torch.cat([visual_mask, sensor_mask[:, :2]], dim=1)
        phases = self.phase_tokens(
            visual_features,
            skeleton_features,
            imu_features,
            availability,
        )
        output = base_logits.clone()
        expert_logits = []
        for expert_index, (group, encoder, normalization, head) in enumerate(
            zip(LABEL_GROUPS, self.expert_encoders, self.expert_normalizations, self.expert_heads)
        ):
            class_token = self.expert_class_tokens[expert_index].expand(video.shape[0], -1, -1)
            encoded = encoder(torch.cat([class_token, phases], dim=1))
            residual = head(normalization(encoded[:, 0]))
            group_indices = torch.tensor(group, device=video.device)
            output[:, group_indices] = (
                output[:, group_indices] + self.residual_scale * residual[:, :-1]
            )
            outside_mask = torch.ones(40, dtype=torch.bool, device=video.device)
            outside_mask[group_indices] = False
            other_logit = torch.logsumexp(base_logits[:, outside_mask], dim=1, keepdim=True)
            expert_logits.append(
                torch.cat(
                    [output[:, group_indices], other_logit + self.residual_scale * residual[:, -1:]],
                    dim=1,
                )
            )
        details = {"base_logits": base_logits, "expert_logits": expert_logits}
        return output, details

    def adapter_parameter_count(self):
        return sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if not name.startswith("base.")
        )


def build_from_source(
    source_path,
    dimension=384,
    phase_depth=1,
    expert_depth=2,
    heads=6,
    feedforward_multiplier=3,
    residual_scale=0.25,
    phases=16,
):
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if config["mode"] != "shared_multimodal" or config.get("use_radar", False):
        raise ValueError("V95 requires a shared no-radar source")
    if config.get("visual_strategy") != "joint":
        raise ValueError("V95 requires joint visual encoding")
    base = build_base_model(config)
    base.load_state_dict(checkpoint_state(checkpoint), strict=True)
    return (
        PhaseAlignedLabelExperts(
            base,
            dimension=dimension,
            phase_depth=phase_depth,
            expert_depth=expert_depth,
            heads=heads,
            feedforward_multiplier=feedforward_multiplier,
            residual_scale=residual_scale,
            phases=phases,
        ),
        config,
    )

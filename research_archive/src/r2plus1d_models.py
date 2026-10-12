import copy
import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional
from torchvision.models.video.resnet import BasicBlock, Conv2Plus1D, R2Plus1dStem, VideoResNet

from src.scratch_models import (
    ConvLSTMTokens,
    ResidualInvariantSkeletonTokens,
    SkeletonTokens,
    StatisticalTokens,
    TemporalTokens,
)


KINETICS_SHA256 = "ade133f18548b59da3a36d6a70d44a4a7cc4b90460654c338cb8a023e9a3e0e1"


class DeviceStatisticalTokens(nn.Module):
    def __init__(self, dimension=192):
        super().__init__()
        self.project = nn.Sequential(
            nn.LayerNorm(9 * 18),
            nn.Linear(9 * 18, 256),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(256, dimension),
            nn.LayerNorm(dimension),
            nn.GELU(),
        )
        self.device_embedding = nn.Parameter(torch.zeros(1, 5, dimension))
        nn.init.trunc_normal_(self.device_embedding, std=0.02)

    def forward(self, values):
        physical = values[:, :, :45].reshape(*values.shape[:2], 5, 9)
        delta = torch.diff(physical, dim=1)
        pooled = [
            physical.mean(1),
            physical.std(1, unbiased=False),
            physical.amax(1),
            physical.amin(1),
            delta.abs().mean(1),
            delta.std(1, unbiased=False),
        ]
        for segments in (2, 4):
            for segment in torch.tensor_split(physical, segments, dim=1):
                pooled.extend([segment.mean(1), segment.std(1, unbiased=False)])
        return self.project(torch.cat(pooled, dim=-1)) + self.device_embedding


class RadarPointTokens(nn.Module):
    def __init__(self, width=96, outputs=192):
        super().__init__()
        self.points = nn.Sequential(
            nn.Conv2d(10, width, 1, bias=False),
            nn.GroupNorm(8, width),
            nn.SiLU(),
            nn.Conv2d(width, width, 1, bias=False),
            nn.GroupNorm(8, width),
            nn.SiLU(),
        )
        self.temporal = TemporalTokens(width * 2, width=128, outputs=outputs)

    def forward(self, values):
        if values.ndim != 4 or values.shape[-1] != 10:
            raise ValueError("Point radar input must have shape [batch, time, points, 10]")
        valid = values[..., 9] > 0.5
        features = self.points(values.permute(0, 3, 1, 2))
        valid_channels = valid.unsqueeze(1)
        maximum = features.masked_fill(~valid_channels, -1e4).amax(dim=3)
        maximum = torch.where(valid.any(dim=2).unsqueeze(1), maximum, 0)
        mean = (features * valid_channels).sum(dim=3) / valid_channels.sum(dim=3).clamp_min(1)
        frames = torch.cat([maximum, mean], dim=1).transpose(1, 2)
        return self.temporal(frames)


def build_r2plus1d34(num_classes=400):
    model = VideoResNet(
        block=BasicBlock,
        conv_makers=[Conv2Plus1D] * 4,
        layers=[3, 4, 6, 3],
        stem=R2Plus1dStem,
        num_classes=num_classes,
    )
    model.layer2[0].conv2[0] = Conv2Plus1D(128, 128, 288)
    model.layer3[0].conv2[0] = Conv2Plus1D(256, 256, 576)
    model.layer4[0].conv2[0] = Conv2Plus1D(512, 512, 1152)
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm3d):
            layer.eps = 1e-3
            layer.momentum = 0.9
    return model


def adapt_input_convolution(convolution, input_channels):
    if convolution.in_channels == input_channels:
        return convolution
    replacement = type(convolution)(
        input_channels,
        convolution.out_channels,
        convolution.kernel_size,
        convolution.stride,
        convolution.padding,
        bias=convolution.bias is not None,
    )
    with torch.no_grad():
        weights = convolution.weight
        copied = min(input_channels, weights.shape[1])
        replacement.weight[:, :copied].copy_(weights[:, :copied])
        if input_channels > copied:
            replacement.weight[:, copied:].copy_(
                weights.mean(dim=1, keepdim=True).expand(-1, input_channels - copied, -1, -1, -1)
            )
        if convolution.bias is not None:
            replacement.bias.copy_(convolution.bias)
    return replacement


def load_kinetics_weights(model, path):
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != KINETICS_SHA256:
        raise ValueError(f"Unexpected Kinetics checkpoint SHA256: {digest}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return digest


class ModalityBatchNorm3d(nn.Module):
    def __init__(self, source, modalities=3):
        super().__init__()
        self.norms = nn.ModuleList([copy.deepcopy(source) for _ in range(modalities)])
        self.active_modality = 0

    def forward(self, values):
        return self.norms[self.active_modality](values)


def replace_modality_batch_norms(module, modalities=3):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm3d):
            setattr(module, name, ModalityBatchNorm3d(child, modalities))
        else:
            replace_modality_batch_norms(child, modalities)


class VideoEncoder(nn.Module):
    def __init__(self, input_channels=4, weights=None, mixstyle_probability=0.0, mixstyle_alpha=0.1,
                 modality_specific_bn=False, separate_stems=False, modality_film=False):
        super().__init__()
        self.mixstyle_probability = mixstyle_probability
        self.mixstyle_alpha = mixstyle_alpha
        self.modality_specific_bn = modality_specific_bn
        self.separate_stems = separate_stems
        self.modality_film = modality_film
        backbone = build_r2plus1d34()
        if weights is not None:
            load_kinetics_weights(backbone, weights)
        backbone.stem[0] = adapt_input_convolution(backbone.stem[0], input_channels)
        self.stem = backbone.stem
        self.extra_stems = nn.ModuleList(
            [copy.deepcopy(backbone.stem) for _ in range(2)] if separate_stems else []
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.modality_affines = nn.ParameterList(
            [nn.Parameter(torch.zeros(3, 2, channels)) for channels in (64, 64, 128, 256, 512)]
            if modality_film else []
        )
        if modality_specific_bn:
            replace_modality_batch_norms(self)

    def mixstyle(self, values):
        if not self.training or self.mixstyle_probability == 0 or torch.rand(()) >= self.mixstyle_probability:
            return values
        mean = values.mean(dim=(2, 3, 4), keepdim=True)
        standard_deviation = values.var(dim=(2, 3, 4), keepdim=True, unbiased=False).add(1e-6).sqrt()
        normalized = (values - mean.detach()) / standard_deviation.detach()
        permutation = torch.randperm(len(values), device=values.device)
        coefficient = torch.distributions.Beta(self.mixstyle_alpha, self.mixstyle_alpha).sample(
            (len(values), 1, 1, 1, 1)
        ).to(values.device)
        mixed_mean = coefficient * mean + (1 - coefficient) * mean[permutation]
        mixed_standard_deviation = (
            coefficient * standard_deviation + (1 - coefficient) * standard_deviation[permutation]
        )
        return normalized * mixed_standard_deviation.detach() + mixed_mean.detach()

    def set_modality(self, modality):
        for module in self.modules():
            if isinstance(module, ModalityBatchNorm3d):
                module.active_modality = modality

    def apply_modality_affine(self, values, stage):
        parameters = self.modality_affines[stage]
        streams = values.chunk(3, dim=0)
        output = []
        for modality, stream in enumerate(streams):
            scale, bias = parameters[modality]
            output.append(
                stream * (1 + scale[None, :, None, None, None])
                + bias[None, :, None, None, None]
            )
        return torch.cat(output, dim=0)

    def forward_trunk(
        self, values, return_multiscale=False, joint_modalities=False, modality=None
    ):
        values = self.layer1(values)
        if joint_modalities and self.modality_film:
            values = self.apply_modality_affine(values, 1)
        elif modality is not None and self.modality_film:
            values = self.apply_single_modality_affine(values, 1, modality)
        values = self.mixstyle(values)
        values = self.layer2(values)
        if joint_modalities and self.modality_film:
            values = self.apply_modality_affine(values, 2)
        elif modality is not None and self.modality_film:
            values = self.apply_single_modality_affine(values, 2, modality)
        intermediate = self.layer3(values)
        if joint_modalities and self.modality_film:
            intermediate = self.apply_modality_affine(intermediate, 3)
        elif modality is not None and self.modality_film:
            intermediate = self.apply_single_modality_affine(intermediate, 3, modality)
        output = self.layer4(intermediate)
        if joint_modalities and self.modality_film:
            output = self.apply_modality_affine(output, 4)
        elif modality is not None and self.modality_film:
            output = self.apply_single_modality_affine(output, 4, modality)
        return (intermediate, output) if return_multiscale else output

    def apply_single_modality_affine(self, values, stage, modality):
        scale, bias = self.modality_affines[stage][modality]
        return (
            values * (1 + scale[None, :, None, None, None])
            + bias[None, :, None, None, None]
        )

    def forward(self, video, modality=0, return_multiscale=False):
        if self.modality_specific_bn:
            self.set_modality(modality)
        values = self.stem(video)
        if self.modality_film:
            values = self.apply_single_modality_affine(values, 0, modality)
        return self.forward_trunk(
            values,
            return_multiscale=return_multiscale,
            modality=modality if self.modality_film else None,
        )

    def forward_joint(self, videos, return_multiscale=False):
        if len(videos) != 3:
            raise ValueError("Joint visual encoding requires exactly three modalities")
        if self.modality_specific_bn:
            raise ValueError("Joint encoding and modality-specific BatchNorm are mutually exclusive")
        if self.separate_stems:
            values = [self.stem(videos[0])]
            values.extend(stem(video) for stem, video in zip(self.extra_stems, videos[1:]))
            values = torch.cat(values, dim=0)
        else:
            values = self.stem(torch.cat(videos, dim=0))
        if self.modality_film:
            values = self.apply_modality_affine(values, 0)
        encoded = self.forward_trunk(
            values,
            return_multiscale=return_multiscale,
            joint_modalities=True,
        )
        if return_multiscale:
            intermediate, output = encoded
            return list(zip(intermediate.chunk(3, dim=0), output.chunk(3, dim=0)))
        return list(encoded.chunk(3, dim=0))


class R2Plus1DClassifier(nn.Module):
    def __init__(self, mode="visual", weights=None, dropout=0.3, dimension=192, fusion_depth=2,
                 input_channels=4, mask_joint_visual=False, use_radar=False,
                 skeleton_representation="coordinates", mixstyle_probability=0.0,
                 mixstyle_alpha=0.1, use_temporal_embedding=False, fusion_output="residual",
                 visual_motion_channels=False, visual_strategy="sequential",
                 use_phase_temporal_embedding=False, per_visual_auxiliary=False,
                 environment_moe=False, visual_multiscale_tokens=False,
                 normalize_modality_tokens=False, skeleton_encoder="tcn",
                 radar_representation="statistics", imu_inputs=85,
                 imu_statistics_residual=False, imu_statistics_logit_residual=False,
                 imu_device_attention_residual=False,
                 imu_statistics_inputs=None):
        super().__init__()
        if mode not in ("visual", "multimodal", "shared_multimodal"):
            raise ValueError(f"Unknown mode: {mode}")
        if fusion_output not in ("residual", "gated", "stream_gated"):
            raise ValueError(f"Unknown fusion output: {fusion_output}")
        if fusion_output == "stream_gated" and mode != "shared_multimodal":
            raise ValueError("Stream-gated fusion requires shared multimodal inputs")
        if visual_strategy not in (
            "sequential", "joint", "modality_bn", "separate_stem_joint", "joint_film",
            "modality_bn_film",
        ):
            raise ValueError(f"Unknown visual strategy: {visual_strategy}")
        if visual_strategy != "sequential" and mode != "shared_multimodal":
            raise ValueError("Alternative visual strategies require shared multimodal inputs")
        if skeleton_encoder not in ("tcn", "residual_invariant"):
            raise ValueError(f"Unknown skeleton encoder: {skeleton_encoder}")
        if skeleton_encoder == "residual_invariant" and skeleton_representation != "coordinates":
            raise ValueError("Residual-invariant skeleton encoding requires coordinate inputs")
        self.mode = mode
        self.shared_visual = mode == "shared_multimodal"
        self.mask_joint_visual = mask_joint_visual
        self.use_radar = use_radar
        self.use_temporal_embedding = use_temporal_embedding
        self.use_phase_temporal_embedding = use_phase_temporal_embedding
        self.fusion_output = fusion_output
        self.visual_motion_channels = visual_motion_channels
        self.visual_strategy = visual_strategy
        self.per_visual_auxiliary = per_visual_auxiliary
        self.environment_moe = environment_moe
        self.visual_multiscale_tokens = visual_multiscale_tokens
        self.normalize_modality_tokens = normalize_modality_tokens
        self.imu_statistics_residual = imu_statistics_residual
        self.imu_statistics_logit_residual = imu_statistics_logit_residual
        self.imu_device_attention_residual = imu_device_attention_residual
        if sum((imu_statistics_residual, imu_statistics_logit_residual, imu_device_attention_residual)) > 1:
            raise ValueError("Choose only one IMU statistics residual route")
        if imu_device_attention_residual and (imu_inputs != 135 or imu_statistics_inputs != 50):
            raise ValueError("Device attention requires the world_official IMU representation")
        self.imu_statistics_start = imu_inputs - (imu_statistics_inputs or imu_inputs)
        if visual_multiscale_tokens and mode != "shared_multimodal":
            raise ValueError("Visual multiscale tokens require shared multimodal inputs")
        if use_radar and mode == "visual":
            raise ValueError("Radar requires a multimodal mode")
        if radar_representation not in ("statistics", "points"):
            raise ValueError(f"Unknown radar representation: {radar_representation}")
        self.video = VideoEncoder(
            2 * (3 if self.shared_visual else input_channels)
            if visual_motion_channels else (3 if self.shared_visual else input_channels),
            weights,
            mixstyle_probability=mixstyle_probability,
            mixstyle_alpha=mixstyle_alpha,
            modality_specific_bn=visual_strategy in ("modality_bn", "modality_bn_film"),
            separate_stems=visual_strategy == "separate_stem_joint",
            modality_film=visual_strategy in ("joint_film", "modality_bn_film"),
        )
        self.video_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, 40))
        if mode == "visual":
            return
        self.skeleton = (
            ResidualInvariantSkeletonTokens()
            if skeleton_encoder == "residual_invariant"
            else SkeletonTokens(skeleton_representation)
        )
        self.imu = ConvLSTMTokens(imu_inputs)
        if imu_statistics_residual or imu_statistics_logit_residual:
            self.imu_statistics = StatisticalTokens(imu_statistics_inputs or imu_inputs)
        if imu_statistics_residual:
            self.imu_statistics_scale = nn.Parameter(torch.zeros(()))
        if imu_statistics_logit_residual:
            self.imu_statistics_head = nn.Linear(dimension, 40)
            self.imu_statistics_logit_scale = nn.Parameter(torch.zeros(()))
        if imu_device_attention_residual:
            self.imu_device_statistics = DeviceStatisticalTokens(dimension)
            self.imu_device_attention = nn.MultiheadAttention(
                dimension, 6, dropout=0.1, batch_first=True
            )
            self.imu_device_normalization = nn.LayerNorm(dimension)
            self.imu_device_head = nn.Linear(dimension, 40)
            nn.init.zeros_(self.imu_device_head.weight)
            nn.init.zeros_(self.imu_device_head.bias)
        if use_radar:
            self.radar = (
                RadarPointTokens()
                if radar_representation == "points"
                else TemporalTokens(28)
            )
        self.video_projection = nn.Linear(512, dimension)
        if visual_multiscale_tokens:
            self.video_intermediate_projection = nn.Linear(256, dimension)
        self.class_token = nn.Parameter(torch.zeros(1, 1, dimension))
        modality_count = (5 if self.shared_visual else 3) + int(use_radar)
        self.modality_embedding = nn.Parameter(torch.zeros(1, modality_count, 1, dimension))
        if normalize_modality_tokens:
            self.modality_normalizations = nn.ModuleList(
                [nn.LayerNorm(dimension) for _ in range(modality_count)]
            )
        if use_temporal_embedding and use_phase_temporal_embedding:
            raise ValueError("Index and phase temporal embeddings are mutually exclusive")
        if use_temporal_embedding or use_phase_temporal_embedding:
            self.temporal_embedding = nn.Parameter(torch.zeros(1, 16, dimension))
        encoder_layer = nn.TransformerEncoderLayer(
            dimension,
            nhead=6,
            dim_feedforward=dimension * 3,
            dropout=0.15,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(encoder_layer, num_layers=fusion_depth, enable_nested_tensor=False)
        self.normalization = nn.LayerNorm(dimension)
        self.head = nn.Sequential(
            nn.Linear(dimension, dimension), nn.GELU(), nn.Dropout(dropout), nn.Linear(dimension, 40)
        )
        if environment_moe:
            self.environment_gate = nn.Linear(dimension, 2)
            self.environment_residual = nn.Linear(dimension, 80)
            nn.init.zeros_(self.environment_residual.weight)
            nn.init.zeros_(self.environment_residual.bias)
        self.skeleton_head = nn.Linear(dimension, 40)
        self.imu_head = nn.Linear(dimension, 40)
        if use_radar:
            self.radar_head = nn.Linear(dimension, 40)
        if fusion_output in ("gated", "stream_gated"):
            expert_count = (
                5 + int(use_radar) if fusion_output == "stream_gated" else 3 + int(use_radar)
            )
            self.expert_gate = nn.Linear(dimension, expert_count * 40)
            nn.init.zeros_(self.expert_gate.weight)
            with torch.no_grad():
                gate_bias = self.expert_gate.bias.reshape(expert_count, 40)
                gate_bias.fill_(-1)
                visual_experts = 3 if fusion_output == "stream_gated" else 1
                gate_bias[:visual_experts].fill_(2)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.modality_embedding, std=0.02)
        if use_temporal_embedding or use_phase_temporal_embedding:
            nn.init.trunc_normal_(self.temporal_embedding, std=0.02)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def visual_representation(self, video):
        if not self.visual_motion_channels:
            return video
        motion = torch.diff(video, dim=2, prepend=video[:, :, :1])
        return torch.cat([video, motion], dim=1)

    def forward(self, video, skeleton=None, imu=None, radar=None, sensor_mask=None, visual_mask=None):
        if self.shared_visual:
            if video.shape[1] != 7:
                raise ValueError("Shared visual mode requires Depth, Thermal, and IR channels")
            visual_inputs = [video[:, :3], video[:, 3:6], video[:, 6:7].expand(-1, 3, -1, -1, -1)]
            visual_inputs = [self.visual_representation(values) for values in visual_inputs]
            if self.visual_strategy in ("joint", "separate_stem_joint", "joint_film"):
                encoded_visual = self.video.forward_joint(
                    visual_inputs, return_multiscale=self.visual_multiscale_tokens
                )
            else:
                encoded_visual = [
                    self.video(
                        values,
                        modality=index,
                        return_multiscale=self.visual_multiscale_tokens,
                    )
                    for index, values in enumerate(visual_inputs)
                ]
            if self.visual_multiscale_tokens:
                video_intermediate_maps = [values[0] for values in encoded_visual]
                video_maps = [values[1] for values in encoded_visual]
            else:
                video_maps = encoded_visual
            video_features = torch.stack([values.mean(dim=(2, 3, 4)) for values in video_maps], dim=1)
            stream_logits = self.video_head(video_features)
            if visual_mask is None:
                visual_mask = torch.ones(video.shape[0], 3, device=video.device)
            weights = visual_mask / visual_mask.sum(dim=1, keepdim=True).clamp_min(1)
            video_logits = (stream_logits * weights.unsqueeze(-1)).sum(dim=1)
        else:
            video_map = self.video(self.visual_representation(video))
            video_features = video_map.mean(dim=(2, 3, 4))
            video_logits = self.video_head(video_features)
            if self.mask_joint_visual and visual_mask is not None:
                video_available = visual_mask.bool().any(dim=1)
                video_logits = video_logits * video_available.unsqueeze(1)
        if self.mode == "visual":
            return video_logits, {"video": video_logits}

        if skeleton is None or imu is None or sensor_mask is None:
            raise ValueError("Multimodal mode requires skeleton, IMU, and masks")
        skeleton_tokens = self.skeleton(skeleton)
        imu_tokens = self.imu(imu)
        imu_statistics = (
            self.imu_statistics(
                imu[:, :, self.imu_statistics_start:]
            )
            if self.imu_statistics_residual or self.imu_statistics_logit_residual
            else None
        )
        if self.imu_statistics_residual:
            imu_tokens = imu_tokens + self.imu_statistics_scale * imu_statistics
        radar_tokens = self.radar(radar) if self.use_radar else None
        if self.shared_visual:
            video_streams = []
            for index, values in enumerate(video_maps):
                final_tokens = self.video_projection(values.mean(dim=(3, 4)).transpose(1, 2))
                if self.visual_multiscale_tokens:
                    intermediate_tokens = self.video_intermediate_projection(
                        video_intermediate_maps[index].mean(dim=(3, 4)).transpose(1, 2)
                    )
                    final_tokens = torch.cat([intermediate_tokens, final_tokens], dim=1)
                video_streams.append(final_tokens)
            streams = video_streams + [skeleton_tokens, imu_tokens]
        else:
            video_tokens = self.video_projection(video_map.mean(dim=(3, 4)).transpose(1, 2))
            streams = [video_tokens, skeleton_tokens, imu_tokens]
        if radar_tokens is not None:
            streams.append(radar_tokens)
        for index, tokens in enumerate(streams):
            if self.normalize_modality_tokens:
                tokens = self.modality_normalizations[index](tokens)
            streams[index] = tokens + self.modality_embedding[:, index]
            if self.use_phase_temporal_embedding:
                phase_embedding = functional.interpolate(
                    self.temporal_embedding.transpose(1, 2),
                    size=tokens.shape[1],
                    mode="linear",
                    align_corners=True,
                ).transpose(1, 2)
                streams[index] = streams[index] + phase_embedding
            elif self.use_temporal_embedding:
                streams[index] = streams[index] + self.temporal_embedding[:, :tokens.shape[1]]
        class_token = self.class_token.expand(video.shape[0], -1, -1)
        tokens = torch.cat([class_token] + streams, dim=1)
        if self.shared_visual:
            video_masks = [
                (~visual_mask[:, index].bool()).unsqueeze(1).expand(-1, tokens.shape[1])
                for index, tokens in enumerate(video_streams)
            ]
        else:
            if self.mask_joint_visual and visual_mask is not None:
                video_missing = ~visual_mask.bool().any(dim=1)
                video_masks = [video_missing.unsqueeze(1).expand(-1, video_tokens.shape[1])]
            else:
                video_masks = [
                    torch.zeros(video.shape[0], video_tokens.shape[1], dtype=torch.bool, device=video.device)
                ]
        skeleton_mask = (~sensor_mask[:, 0].bool()).unsqueeze(1).expand(-1, skeleton_tokens.shape[1])
        imu_mask = (~sensor_mask[:, 1].bool()).unsqueeze(1).expand(-1, imu_tokens.shape[1])
        radar_masks = (
            [(~sensor_mask[:, 2].bool()).unsqueeze(1).expand(-1, radar_tokens.shape[1])]
            if radar_tokens is not None else []
        )
        padding_mask = torch.cat(
            [torch.zeros(video.shape[0], 1, dtype=torch.bool, device=video.device)] + video_masks
            + [skeleton_mask, imu_mask] + radar_masks,
            dim=1,
        )
        fused = self.fusion(tokens, src_key_padding_mask=padding_mask)
        fused_class = self.normalization(fused[:, 0])
        auxiliary = {
            ("video_streams" if self.shared_visual and self.per_visual_auxiliary else "video"):
                stream_logits if self.shared_visual and self.per_visual_auxiliary else video_logits,
            "skeleton": self.skeleton_head(skeleton_tokens.mean(dim=1)),
            "imu": self.imu_head(imu_tokens.mean(dim=1)),
        }
        if radar_tokens is not None:
            auxiliary["radar"] = self.radar_head(radar_tokens.mean(dim=1))
        if self.fusion_output in ("gated", "stream_gated"):
            expert_logits = (
                torch.cat(
                    [stream_logits]
                    + [auxiliary[name].unsqueeze(1) for name in ("skeleton", "imu")]
                    + ([auxiliary["radar"].unsqueeze(1)] if radar_tokens is not None else []),
                    dim=1,
                )
                if self.fusion_output == "stream_gated"
                else torch.stack(list(auxiliary.values()), dim=1)
            )
            gate_logits = self.expert_gate(fused_class).reshape(
                video.shape[0], expert_logits.shape[1], 40
            )
            if self.fusion_output == "stream_gated":
                expert_available = torch.cat([visual_mask.bool(), sensor_mask.bool()], dim=1)
            else:
                video_available = (
                    visual_mask.bool().any(dim=1)
                    if visual_mask is not None
                    else torch.ones(video.shape[0], dtype=torch.bool, device=video.device)
                )
                expert_available = torch.cat([video_available.unsqueeze(1), sensor_mask.bool()], dim=1)
            gate_logits = gate_logits.masked_fill(~expert_available.unsqueeze(2), -1e4)
            output = (gate_logits.softmax(dim=1) * expert_logits).sum(dim=1) + self.head(fused_class)
        else:
            output = video_logits + self.head(fused_class)
        if self.imu_statistics_logit_residual:
            output = output + self.imu_statistics_logit_scale * self.imu_statistics_head(
                imu_statistics.mean(dim=1)
            )
        if self.imu_device_attention_residual:
            device_tokens = self.imu_device_statistics(
                imu[:, :, self.imu_statistics_start:]
            )
            device_missing = imu[:, 0, -5:] <= 0
            all_missing = device_missing.all(dim=1)
            if all_missing.any():
                device_missing = device_missing.clone()
                device_missing[all_missing, 0] = False
                device_tokens = device_tokens.clone()
                device_tokens[all_missing, 0] = 0
            device_context, _ = self.imu_device_attention(
                fused_class.unsqueeze(1),
                device_tokens,
                device_tokens,
                key_padding_mask=device_missing,
                need_weights=False,
            )
            output = output + self.imu_device_head(
                self.imu_device_normalization(device_context[:, 0])
            )
        if self.environment_moe:
            environment_logits = self.environment_gate(fused_class)
            environment_weights = environment_logits.softmax(dim=1)
            environment_residuals = self.environment_residual(fused_class).reshape(-1, 2, 40)
            output = output + (environment_weights.unsqueeze(2) * environment_residuals).sum(dim=1)
            auxiliary["environment"] = environment_logits
        return output, auxiliary


def quantize_state_dict(state_dict):
    quantized = {}
    tensor_bytes = 0
    for name, value in state_dict.items():
        value = value.detach().cpu()
        if value.is_floating_point() and value.ndim >= 2:
            dimensions = tuple(range(1, value.ndim))
            scale = (value.float().abs().amax(dim=dimensions, keepdim=True) / 127).clamp_min(1e-12)
            integer = (value.float() / scale).round().clamp(-127, 127).to(torch.int8)
            quantized[name] = {"integer": integer, "scale": scale.half()}
            tensor_bytes += integer.numel() + scale.numel() * scale.element_size()
        else:
            stored = value.half() if value.is_floating_point() else value
            quantized[name] = stored
            tensor_bytes += stored.numel() * stored.element_size()
    return quantized, tensor_bytes


def quantize_state_dict_int4(state_dict):
    quantized = {}
    tensor_bytes = 0
    for name, value in state_dict.items():
        value = value.detach().cpu()
        if value.is_floating_point() and value.ndim >= 2:
            dimensions = tuple(range(1, value.ndim))
            scale = (value.float().abs().amax(dim=dimensions, keepdim=True) / 7).clamp_min(1e-12)
            integer = (value.float() / scale).round().clamp(-7, 7).to(torch.int8).flatten()
            encoded = (integer.to(torch.int16) + 8).to(torch.uint8)
            if encoded.numel() % 2:
                encoded = torch.cat([encoded, torch.full((1,), 8, dtype=torch.uint8)])
            packed = encoded[0::2] | (encoded[1::2] << 4)
            quantized[name] = {
                "packed": packed,
                "scale": scale.half(),
                "shape": tuple(value.shape),
            }
            tensor_bytes += packed.numel() + scale.numel() * scale.element_size()
        else:
            stored = value.half() if value.is_floating_point() else value
            quantized[name] = stored
            tensor_bytes += stored.numel() * stored.element_size()
    return quantized, tensor_bytes


def quantize_state_dict_int4_grouped(state_dict, group_size=128):
    if group_size < 16 or group_size % 2:
        raise ValueError("INT4 group size must be an even integer of at least 16")
    quantized = {}
    tensor_bytes = 0
    for name, value in state_dict.items():
        value = value.detach().cpu()
        if value.is_floating_point() and value.ndim >= 2:
            matrix = value.float().reshape(value.shape[0], -1)
            groups = (matrix.shape[1] + group_size - 1) // group_size
            padding = groups * group_size - matrix.shape[1]
            if padding:
                matrix = functional.pad(matrix, (0, padding))
            grouped = matrix.reshape(matrix.shape[0], groups, group_size)
            scale = (grouped.abs().amax(dim=2, keepdim=True) / 7).clamp_min(1e-12)
            integer = (grouped / scale).round().clamp(-7, 7).to(torch.int8)
            integer = integer.reshape(matrix.shape[0], -1)[:, :value[0].numel()].flatten()
            encoded = (integer.to(torch.int16) + 8).to(torch.uint8)
            if encoded.numel() % 2:
                encoded = torch.cat([encoded, torch.full((1,), 8, dtype=torch.uint8)])
            packed = encoded[0::2] | (encoded[1::2] << 4)
            quantized[name] = {
                "packed": packed,
                "scale": scale.half(),
                "shape": tuple(value.shape),
                "bits": -group_size,
            }
            tensor_bytes += packed.numel() + scale.numel() * scale.element_size()
        else:
            stored = value.half() if value.is_floating_point() else value
            quantized[name] = stored
            tensor_bytes += stored.numel() * stored.element_size()
    return quantized, tensor_bytes


def quantize_state_dict_nf4_grouped(state_dict, group_size=128):
    if group_size < 16 or group_size % 2:
        raise ValueError("NF4 group size must be an even integer of at least 16")
    codebook = torch.tensor(
        [
            -1.0, -0.6961928, -0.52507305, -0.3949175,
            -0.28444138, -0.18477343, -0.09105004, 0.0,
            0.0795803, 0.1609302, 0.2461123, 0.33791524,
            0.44070983, 0.562617, 0.72295684, 1.0,
        ],
        dtype=torch.float32,
    )
    boundaries = (codebook[:-1] + codebook[1:]) / 2
    quantized = {}
    tensor_bytes = 0
    for name, value in state_dict.items():
        value = value.detach().cpu()
        if value.is_floating_point() and value.ndim >= 2:
            matrix = value.float().reshape(value.shape[0], -1)
            groups = (matrix.shape[1] + group_size - 1) // group_size
            padding = groups * group_size - matrix.shape[1]
            if padding:
                matrix = functional.pad(matrix, (0, padding))
            grouped = matrix.reshape(matrix.shape[0], groups, group_size)
            scale = grouped.abs().amax(dim=2, keepdim=True).clamp_min(1e-12)
            encoded = torch.bucketize(grouped / scale, boundaries).to(torch.uint8)
            encoded = encoded.reshape(matrix.shape[0], -1)[:, :value[0].numel()].flatten()
            if encoded.numel() % 2:
                encoded = torch.cat([encoded, torch.full((1,), 7, dtype=torch.uint8)])
            packed = encoded[0::2] | (encoded[1::2] << 4)
            quantized[name] = {
                "packed": packed,
                "scale": scale.half(),
                "shape": tuple(value.shape),
                "bits": -(10_000 + group_size),
            }
            tensor_bytes += packed.numel() + scale.numel() * scale.element_size()
        else:
            stored = value.half() if value.is_floating_point() else value
            quantized[name] = stored
            tensor_bytes += stored.numel() * stored.element_size()
    return quantized, tensor_bytes


def quantize_state_dict_int6(state_dict):
    quantized = {}
    tensor_bytes = 0
    for name, value in state_dict.items():
        value = value.detach().cpu()
        if value.is_floating_point() and value.ndim >= 2:
            dimensions = tuple(range(1, value.ndim))
            scale = (value.float().abs().amax(dim=dimensions, keepdim=True) / 31).clamp_min(1e-12)
            integer = (value.float() / scale).round().clamp(-31, 31).to(torch.int8).flatten()
            encoded = (integer.to(torch.int16) + 32).to(torch.uint8)
            padding = (-encoded.numel()) % 4
            if padding:
                encoded = torch.cat([encoded, torch.full((padding,), 32, dtype=torch.uint8)])
            values = encoded.reshape(-1, 4)
            packed = torch.empty(values.shape[0], 3, dtype=torch.uint8)
            packed[:, 0] = values[:, 0] | ((values[:, 1] & 3) << 6)
            packed[:, 1] = (values[:, 1] >> 2) | ((values[:, 2] & 15) << 4)
            packed[:, 2] = (values[:, 2] >> 4) | (values[:, 3] << 2)
            quantized[name] = {
                "packed": packed.flatten(),
                "scale": scale.half(),
                "shape": tuple(value.shape),
                "bits": 6,
            }
            tensor_bytes += packed.numel() + scale.numel() * scale.element_size()
        else:
            stored = value.half() if value.is_floating_point() else value
            quantized[name] = stored
            tensor_bytes += stored.numel() * stored.element_size()
    return quantized, tensor_bytes


def quantize_state_dict_int7(state_dict):
    quantized = {}
    tensor_bytes = 0
    for name, value in state_dict.items():
        value = value.detach().cpu()
        if value.is_floating_point() and value.ndim >= 2:
            dimensions = tuple(range(1, value.ndim))
            scale = (value.float().abs().amax(dim=dimensions, keepdim=True) / 63).clamp_min(1e-12)
            integer = (value.float() / scale).round().clamp(-63, 63).to(torch.int8).flatten()
            encoded = (integer.to(torch.int16) + 64).to(torch.uint8)
            padding = (-encoded.numel()) % 8
            if padding:
                encoded = torch.cat([encoded, torch.full((padding,), 64, dtype=torch.uint8)])
            values = encoded.reshape(-1, 8)
            packed = torch.empty(values.shape[0], 7, dtype=torch.uint8)
            packed[:, 0] = values[:, 0] | ((values[:, 1] & 1) << 7)
            packed[:, 1] = (values[:, 1] >> 1) | ((values[:, 2] & 3) << 6)
            packed[:, 2] = (values[:, 2] >> 2) | ((values[:, 3] & 7) << 5)
            packed[:, 3] = (values[:, 3] >> 3) | ((values[:, 4] & 15) << 4)
            packed[:, 4] = (values[:, 4] >> 4) | ((values[:, 5] & 31) << 3)
            packed[:, 5] = (values[:, 5] >> 5) | ((values[:, 6] & 63) << 2)
            packed[:, 6] = (values[:, 6] >> 6) | (values[:, 7] << 1)
            quantized[name] = {
                "packed": packed.flatten(),
                "scale": scale.half(),
                "shape": tuple(value.shape),
                "bits": 7,
            }
            tensor_bytes += packed.numel() + scale.numel() * scale.element_size()
        else:
            stored = value.half() if value.is_floating_point() else value
            quantized[name] = stored
            tensor_bytes += stored.numel() * stored.element_size()
    return quantized, tensor_bytes


def compact_quantized_state_dict(state_dict):
    grouped = {}
    entries = []

    def append(tensor):
        tensor = tensor.detach().cpu().contiguous().flatten()
        key = str(tensor.dtype).removeprefix("torch.")
        offset = sum(value.numel() for value in grouped.setdefault(key, []))
        grouped[key].append(tensor)
        return key, offset, tensor.numel()

    for name, value in state_dict.items():
        if isinstance(value, dict) and "integer" in value:
            value_key, value_offset, value_count = append(value["integer"])
            scale_key, scale_offset, scale_count = append(value["scale"])
            entries.append((name, "integer", value_key, value_offset, value_count,
                            tuple(value["integer"].shape), scale_key, scale_offset,
                            scale_count, tuple(value["scale"].shape), 8))
        elif isinstance(value, dict) and "packed" in value:
            value_key, value_offset, value_count = append(value["packed"])
            scale_key, scale_offset, scale_count = append(value["scale"])
            entries.append((name, "packed", value_key, value_offset, value_count,
                            tuple(value["shape"]), scale_key, scale_offset,
                            scale_count, tuple(value["scale"].shape), value.get("bits", 4)))
        else:
            value_key, value_offset, value_count = append(value)
            entries.append((name, "tensor", value_key, value_offset, value_count,
                            tuple(value.shape), "", 0, 0, (), 0))
    buffers = {key: torch.cat(values) for key, values in grouped.items()}
    return {"format": "compact_quantized_state_dict_v1", "entries": entries, "buffers": buffers}


def expand_compact_quantized_state_dict(state_dict):
    if state_dict.get("format") != "compact_quantized_state_dict_v1":
        return state_dict
    output = {}
    buffers = state_dict["buffers"]
    for (name, kind, value_key, value_offset, value_count, shape, scale_key,
         scale_offset, scale_count, scale_shape, bits) in state_dict["entries"]:
        stored = buffers[value_key][value_offset:value_offset + value_count]
        if kind == "tensor":
            output[name] = stored.reshape(shape)
            continue
        scale = buffers[scale_key][scale_offset:scale_offset + scale_count].reshape(scale_shape)
        if kind == "integer":
            output[name] = {"integer": stored.reshape(shape), "scale": scale}
        else:
            output[name] = {"packed": stored, "scale": scale, "shape": shape, "bits": bits}
    return output


def dequantize_state_dict(state_dict):
    state_dict = expand_compact_quantized_state_dict(state_dict)
    output = {}
    for name, value in state_dict.items():
        if isinstance(value, dict) and "integer" in value:
            output[name] = value["integer"].float() * value["scale"].float()
        elif isinstance(value, dict) and "packed" in value and value.get("bits", 4) == 4:
            packed = value["packed"]
            encoded = torch.empty(packed.numel() * 2, dtype=torch.uint8)
            encoded[0::2] = packed & 15
            encoded[1::2] = packed >> 4
            count = 1
            for size in value["shape"]:
                count *= size
            integer = encoded[:count].to(torch.int16).sub_(8).reshape(value["shape"])
            output[name] = integer.float() * value["scale"].float()
        elif isinstance(value, dict) and "packed" in value and value.get("bits", 0) <= -10_000:
            group_size = -int(value["bits"]) - 10_000
            packed = value["packed"]
            encoded = torch.empty(packed.numel() * 2, dtype=torch.uint8)
            encoded[0::2] = packed & 15
            encoded[1::2] = packed >> 4
            codebook = torch.tensor(
                [
                    -1.0, -0.6961928, -0.52507305, -0.3949175,
                    -0.28444138, -0.18477343, -0.09105004, 0.0,
                    0.0795803, 0.1609302, 0.2461123, 0.33791524,
                    0.44070983, 0.562617, 0.72295684, 1.0,
                ],
                dtype=torch.float32,
            )
            scale = value["scale"].float()
            inner = 1
            for size in value["shape"][1:]:
                inner *= size
            count = value["shape"][0] * inner
            normalized = codebook[encoded[:count].long()].reshape(value["shape"][0], inner)
            padding = scale.shape[1] * group_size - inner
            if padding:
                normalized = functional.pad(normalized, (0, padding))
            grouped = normalized.reshape(*scale.shape[:-1], group_size) * scale
            output[name] = grouped.reshape(value["shape"][0], -1)[:, :inner].reshape(value["shape"])
        elif isinstance(value, dict) and "packed" in value and value.get("bits", 0) < 0:
            group_size = -int(value["bits"])
            packed = value["packed"]
            encoded = torch.empty(packed.numel() * 2, dtype=torch.uint8)
            encoded[0::2] = packed & 15
            encoded[1::2] = packed >> 4
            scale = value["scale"].float()
            inner = 1
            for size in value["shape"][1:]:
                inner *= size
            count = value["shape"][0] * inner
            integer = encoded[:count].to(torch.int16).sub_(8).reshape(value["shape"][0], inner)
            padding = scale.shape[1] * group_size - inner
            if padding:
                integer = functional.pad(integer, (0, padding))
            grouped = integer.reshape(*scale.shape[:-1], group_size).float() * scale
            output[name] = grouped.reshape(value["shape"][0], -1)[:, :inner].reshape(value["shape"])
        elif isinstance(value, dict) and "packed" in value and value.get("bits") == 6:
            packed = value["packed"].reshape(-1, 3)
            encoded = torch.empty(packed.shape[0], 4, dtype=torch.uint8)
            encoded[:, 0] = packed[:, 0] & 63
            encoded[:, 1] = (packed[:, 0] >> 6) | ((packed[:, 1] & 15) << 2)
            encoded[:, 2] = (packed[:, 1] >> 4) | ((packed[:, 2] & 3) << 4)
            encoded[:, 3] = packed[:, 2] >> 2
            count = 1
            for size in value["shape"]:
                count *= size
            integer = encoded.flatten()[:count].to(torch.int16).sub_(32).reshape(value["shape"])
            output[name] = integer.float() * value["scale"].float()
        elif isinstance(value, dict) and "packed" in value and value.get("bits") == 7:
            packed = value["packed"].reshape(-1, 7)
            encoded = torch.empty(packed.shape[0], 8, dtype=torch.uint8)
            encoded[:, 0] = packed[:, 0] & 127
            encoded[:, 1] = (packed[:, 0] >> 7) | ((packed[:, 1] & 63) << 1)
            encoded[:, 2] = (packed[:, 1] >> 6) | ((packed[:, 2] & 31) << 2)
            encoded[:, 3] = (packed[:, 2] >> 5) | ((packed[:, 3] & 15) << 3)
            encoded[:, 4] = (packed[:, 3] >> 4) | ((packed[:, 4] & 7) << 4)
            encoded[:, 5] = (packed[:, 4] >> 3) | ((packed[:, 5] & 3) << 5)
            encoded[:, 6] = (packed[:, 5] >> 2) | ((packed[:, 6] & 1) << 6)
            encoded[:, 7] = packed[:, 6] >> 1
            count = 1
            for size in value["shape"]:
                count *= size
            integer = encoded.flatten()[:count].to(torch.int16).sub_(64).reshape(value["shape"])
            output[name] = integer.float() * value["scale"].float()
        else:
            output[name] = value.float() if value.is_floating_point() else value
    return output


def apply_low_rank_delta(state_dict, adapter):
    if adapter.get("format") != "compact_low_rank_delta_v1":
        raise ValueError("Unsupported low-rank adapter format")
    output = {name: value.clone() for name, value in state_dict.items()}
    integer = adapter["integer"]
    scale = adapter["scale"]
    for entry in adapter["entries"]:
        name = entry["name"]
        if name not in output or tuple(output[name].shape) != tuple(entry["shape"]):
            raise ValueError(f"Low-rank adapter tensor mismatch: {name}")
        left_rows = entry["left_rows"]
        right_rows = entry["right_rows"]
        left = integer[
            entry["left_integer_offset"]:
            entry["left_integer_offset"] + entry["left_integer_count"]
        ].reshape(left_rows, -1).float()
        right = integer[
            entry["right_integer_offset"]:
            entry["right_integer_offset"] + entry["right_integer_count"]
        ].reshape(right_rows, -1).float()
        left_scale = scale[
            entry["left_scale_offset"]:entry["left_scale_offset"] + left_rows
        ].reshape(left_rows, 1).float()
        right_scale = scale[
            entry["right_scale_offset"]:entry["right_scale_offset"] + right_rows
        ].reshape(right_rows, 1).float()
        output[name] = output[name] + (
            (left * left_scale) @ (right * right_scale)
        ).reshape(entry["shape"])
    for name, delta in adapter["exact"].items():
        if name not in output or output[name].shape != delta.shape:
            raise ValueError(f"Low-rank exact tensor mismatch: {name}")
        output[name] = output[name] + delta.to(output[name].dtype) if delta.is_floating_point() else delta
    return output

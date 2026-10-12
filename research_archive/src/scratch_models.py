import torch
from torch import nn
from torch.nn import functional


MODALITIES = ("scene_ir", "thermal", "scene_depth", "skeleton", "imu", "imu_world", "imu_relative",
              "imu_official", "imu_magnetic", "radar")


def validate_scratch_config(config):
    modalities = config["modalities"]
    if isinstance(modalities, str):
        modalities = modalities.split(",")
    if not modalities or len(set(modalities)) != len(modalities) or set(modalities) - set(MODALITIES):
        raise ValueError("Scratch inputs must be unique raw-scene/sensor modalities; no detector crops or pretrained features")
    for name in ("imagenet", "kinetics", "init", "pretrained", "temporal_shift", "freeze_backbone_bn"):
        if config.get(name):
            raise ValueError(f"Scratch protocol forbids {name}")
    for name, expected in (("backbone", "scratch"), ("skeleton_backbone", "tcn"), ("temporal_backbone", "tcn")):
        if config.get(name, expected) != expected:
            raise ValueError(f"Scratch protocol requires {name}={expected}; graph/statistics experts are excluded")
    if config.get("single_modality_mixup", False):
        if len(modalities) != 1 or not 0 < config.get("mixup", 0) <= 1:
            raise ValueError("Explicit single-modality mixup requires one modality and alpha in (0, 1]")
    elif config.get("mixup", 0) != 0:
        raise ValueError("Scratch mixup requires an explicit single-modality experiment to preserve missing masks")
    if config.get("spatial_augment", False):
        raise ValueError("Scratch protocol keeps generic spatial augmentation disabled")
    if config.get("fusion", "concat") not in ("concat", "transformer", "cross_transformer"):
        raise ValueError("Only concatenation classification or Transformer fusion is supported")
    if config.get("visual_multiplier", 1) not in (1, 2):
        raise ValueError("Visual capacity multiplier must be 1 or 2")
    if config.get("visual_encoder", "legacy") not in ("legacy", "cnn_pooled", "cnn_tokens", "patch_tokens"):
        raise ValueError("Unknown visual encoder")
    if config.get("visual_encoder", "legacy") != "legacy":
        if config.get("visual_multiplier", 1) != 1 or config.get("frames", 16) != 16:
            raise ValueError("Spatial study fixes multiplier=1 and frames=16")
        if config.get("image_size", 160) % 16 or config.get("visual_feedforward", 384) < 192:
            raise ValueError("Invalid spatial study image size or feedforward width")
        if config.get("init"):
            raise ValueError("Spatial architecture comparisons do not accept generic warm starts")
    if config.get("skeleton_representation", "coordinates") not in ("coordinates", "centered", "invariant"):
        raise ValueError("Unknown skeleton representation")
    if config.get("scratch_skeleton_encoder", "tcn") not in ("tcn", "dstformer"):
        raise ValueError("Unknown scratch skeleton encoder")
    if config.get("imu_token_encoder", "conv_lstm") not in ("conv_lstm", "statistics", "transformer"):
        raise ValueError("Unknown IMU token encoder")
    if config.get("scratch_skeleton_encoder", "tcn") == "dstformer":
        dimension = config.get("skeleton_dimension", 192)
        depth = config.get("skeleton_depth", 3)
        heads = config.get("skeleton_heads", 6)
        if dimension < 64 or depth < 1 or heads < 1 or dimension % heads:
            raise ValueError("Invalid DSTformer dimension, depth, or attention heads")
        if config.get("skeleton_representation", "coordinates") == "invariant":
            raise ValueError("DSTformer requires joint-wise coordinate streams")
    if config.get("skeleton_representation", "coordinates") != "coordinates" and "skeleton" not in modalities:
        raise ValueError("Skeleton representation requires the skeleton modality")
    for option, default, permitted, modality in [("thermal_view", "full", ["full", "motion_crop"], "thermal"),
                                                ("imu_representation", "raw", ["raw", "world"], "imu")]:
        value = config.get(option, default)
        if value not in permitted or value != default and modality not in modalities:
            raise ValueError(f"Invalid {option} for the selected modalities")
    if config.get("thermal_view", "full") == "motion_crop" and config.get("image_size", 160) != 160:
        raise ValueError("Thermal motion crop requires image_size=160")


class ResidualTemporal(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(width, width, 5, padding=2 * dilation, dilation=dilation, bias=False),
            nn.GroupNorm(8, width), nn.SiLU(), nn.Dropout(0.1),
            nn.Conv1d(width, width, 1, bias=False), nn.GroupNorm(8, width),
        )

    def forward(self, values):
        return functional.silu(values + self.layers(values))


class TemporalTokens(nn.Module):
    def __init__(self, inputs, width=96, outputs=192):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(inputs, width, 3, padding=1, bias=False), nn.GroupNorm(8, width), nn.SiLU(),
            ResidualTemporal(width, 1), ResidualTemporal(width, 2), ResidualTemporal(width, 4),
            nn.Conv1d(width, outputs, 1), nn.AdaptiveAvgPool1d(16),
        )

    def forward(self, values):
        return self.layers(values.transpose(1, 2)).transpose(1, 2)


def skeleton_streams(values, representation):
    if representation == "coordinates":
        return values.split(51, dim=-1)
    pose, bones, velocity = [part.reshape(*values.shape[:2], 17, 3) for part in values.split(51, dim=-1)]
    if representation == "centered":
        return [(pose - pose[:, :, :1]).flatten(2), bones.flatten(2),
                (velocity - velocity[:, :, :1]).flatten(2)]
    if representation != "invariant":
        raise ValueError("Unknown skeleton representation")
    pairs = torch.triu_indices(17, 17, offset=1, device=values.device)
    scale = torch.linalg.vector_norm(pose[:, :, 8] - pose[:, :, 0], dim=-1).median(1).values.clamp_min(1e-4)
    distances = torch.linalg.vector_norm(pose[:, :, pairs[0]] - pose[:, :, pairs[1]], dim=-1) / scale[:, None, None]
    parents = [0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
    directions = functional.normalize(pose - pose[:, :, parents], dim=-1, eps=1e-4)
    angles = (directions[:, :, pairs[0]] * directions[:, :, pairs[1]]).sum(-1)
    motion = torch.diff(distances, dim=1, prepend=distances[:, :1]) * 5
    return [(distances / 3).clamp(-10, 10), angles, motion.clamp(-10, 10)]


class SkeletonTokens(nn.Module):
    def __init__(self, representation="coordinates"):
        super().__init__()
        self.representation = representation
        inputs = 136 if representation == "invariant" else 51
        self.streams = nn.ModuleList([TemporalTokens(inputs, 64, 64) for _ in range(3)])
        self.merge = nn.Conv1d(192, 192, 1)

    def forward(self, values):
        streams = [encoder(part) for encoder, part in zip(self.streams, skeleton_streams(values, self.representation))]
        return self.merge(torch.cat(streams, dim=-1).transpose(1, 2)).transpose(1, 2)


def body_centered_skeleton(values):
    pose = values[:, :, :51].reshape(*values.shape[:2], 17, 3)
    centered_pose = pose - pose[:, :, :1]
    scale = torch.linalg.vector_norm(centered_pose[:, :, 8], dim=-1).median(dim=1).values.clamp(0.15, 1.2)
    centered_pose = centered_pose / scale[:, None, None, None]
    parents = [0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
    bones = centered_pose - centered_pose[:, :, parents]
    velocity = torch.diff(centered_pose, dim=1, prepend=centered_pose[:, :1]) * 10
    return torch.cat(
        [centered_pose.flatten(2), bones.flatten(2), velocity.flatten(2)], dim=-1
    ).clamp(-10, 10)


class ResidualInvariantSkeletonTokens(SkeletonTokens):
    def __init__(self):
        super().__init__("coordinates")
        self.invariant = SkeletonTokens("coordinates")
        self.view_projection = nn.Linear(384, 192, bias=False)
        with torch.no_grad():
            self.view_projection.weight.zero_()
            self.view_projection.weight[:, :192].copy_(torch.eye(192))

    def forward(self, values):
        raw_tokens = super().forward(values)
        invariant_tokens = self.invariant(body_centered_skeleton(values))
        return self.view_projection(torch.cat([raw_tokens, invariant_tokens], dim=-1))


class FactorizedAttentionBlock(nn.Module):
    def __init__(self, dimension, heads, feedforward, dropout, spatial_first):
        super().__init__()
        self.spatial_first = spatial_first
        self.spatial = nn.TransformerEncoderLayer(
            dimension, heads, feedforward, dropout=dropout, batch_first=True, norm_first=True,
        )
        self.temporal = nn.TransformerEncoderLayer(
            dimension, heads, feedforward, dropout=dropout, batch_first=True, norm_first=True,
        )

    def apply_spatial(self, values):
        batch, steps, joints, dimension = values.shape
        return self.spatial(values.reshape(batch * steps, joints, dimension)).reshape(batch, steps, joints, dimension)

    def apply_temporal(self, values):
        batch, steps, joints, dimension = values.shape
        values = values.permute(0, 2, 1, 3).reshape(batch * joints, steps, dimension)
        values = self.temporal(values)
        return values.reshape(batch, joints, steps, dimension).permute(0, 2, 1, 3)

    def forward(self, values):
        if self.spatial_first:
            return self.apply_temporal(self.apply_spatial(values))
        return self.apply_spatial(self.apply_temporal(values))


class DualPathSkeletonTokens(nn.Module):
    def __init__(self, dimension=192, depth=3, heads=6, outputs=192, frames=64):
        super().__init__()
        feedforward = dimension * 3
        self.input_embedding = nn.Linear(9, dimension)
        self.joint_embedding = nn.Parameter(torch.zeros(1, 1, 17, dimension))
        self.time_embedding = nn.Parameter(torch.zeros(1, frames, 1, dimension))
        self.spatial_temporal = nn.ModuleList([
            FactorizedAttentionBlock(dimension, heads, feedforward, 0.1, True) for _ in range(depth)
        ])
        self.temporal_spatial = nn.ModuleList([
            FactorizedAttentionBlock(dimension, heads, feedforward, 0.1, False) for _ in range(depth)
        ])
        self.path_gates = nn.ModuleList([nn.Linear(dimension * 2, 2) for _ in range(depth)])
        self.normalization = nn.LayerNorm(dimension)
        self.joint_attention = nn.Linear(dimension, 1)
        self.output = nn.Linear(dimension, outputs)
        nn.init.trunc_normal_(self.joint_embedding, std=0.02)
        nn.init.trunc_normal_(self.time_embedding, std=0.02)
        for gate in self.path_gates:
            nn.init.zeros_(gate.weight)
            nn.init.zeros_(gate.bias)

    def forward(self, values):
        streams = [part.reshape(*values.shape[:2], 17, 3) for part in values.split(51, dim=-1)]
        values = torch.cat(streams, dim=-1)
        values = self.input_embedding(values)
        values = values + self.joint_embedding + self.time_embedding[:, :values.shape[1]]
        for spatial_temporal, temporal_spatial, gate in zip(
                self.spatial_temporal, self.temporal_spatial, self.path_gates):
            first = spatial_temporal(values)
            second = temporal_spatial(values)
            weights = gate(torch.cat([first, second], dim=-1)).softmax(dim=-1)
            values = first * weights[..., :1] + second * weights[..., 1:]
        values = self.normalization(values)
        joint_weights = self.joint_attention(values).softmax(dim=2)
        values = (values * joint_weights).sum(dim=2)
        values = functional.adaptive_avg_pool1d(values.transpose(1, 2), 16).transpose(1, 2)
        return self.output(values)


class ConvLSTMTokens(nn.Module):
    def __init__(self, inputs=85):
        super().__init__()
        self.convolution = nn.Sequential(
            nn.Conv1d(inputs, 96, 5, padding=2, bias=False), nn.GroupNorm(8, 96), nn.SiLU(),
            nn.Conv1d(96, 96, 5, padding=2, bias=False), nn.GroupNorm(8, 96), nn.SiLU(),
        )
        self.recurrent = nn.LSTM(96, 96, batch_first=True, bidirectional=True)

    def forward(self, values):
        features = self.convolution(values.transpose(1, 2)).transpose(1, 2)
        features, _ = self.recurrent(features)
        return functional.adaptive_avg_pool1d(features.transpose(1, 2), 16).transpose(1, 2)


class StatisticalTokens(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.project = nn.Sequential(
            nn.LayerNorm(inputs * 18),
            nn.Linear(inputs * 18, 512),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(512, 192),
            nn.LayerNorm(192),
            nn.GELU(),
        )

    def forward(self, values):
        delta = torch.diff(values, dim=1)
        pooled = [
            values.mean(1),
            values.std(1, unbiased=False),
            values.amax(1),
            values.amin(1),
            delta.abs().mean(1),
            delta.std(1, unbiased=False),
        ]
        for segments in (2, 4):
            for segment in torch.tensor_split(values, segments, dim=1):
                pooled.extend([segment.mean(1), segment.std(1, unbiased=False)])
        return self.project(torch.cat(pooled, dim=1)).unsqueeze(1)


class TransformerIMUTokens(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.convolution = nn.Sequential(
            nn.Conv1d(inputs, 64, 3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 96, 3, padding=1, bias=False),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Conv1d(96, 128, 3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.position = nn.Parameter(torch.zeros(1, 64, 128))
        layer = nn.TransformerEncoderLayer(
            128, 4, 384, dropout=0.15, activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, 2, norm=nn.LayerNorm(128), enable_nested_tensor=False)
        self.project = nn.Sequential(nn.Flatten(), nn.Dropout(0.35), nn.Linear(64 * 128, 192), nn.GELU())
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, values):
        values = self.convolution(values.transpose(1, 2)).transpose(1, 2)
        return self.project(self.transformer(values + self.position)).unsqueeze(1)


class ResidualSpatial(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(inputs, outputs, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, outputs), nn.SiLU(),
            nn.Conv2d(outputs, outputs, 3, padding=1, bias=False), nn.GroupNorm(8, outputs),
        )
        self.skip = nn.Conv2d(inputs, outputs, 1, stride=2, bias=False)

    def forward(self, values):
        return functional.silu(self.layers(values) + self.skip(values))


class VideoTokens(nn.Module):
    def __init__(self, multiplier=1):
        super().__init__()
        self.frames = nn.Sequential(
            ResidualSpatial(3, 32 * multiplier), ResidualSpatial(32 * multiplier, 64 * multiplier),
            ResidualSpatial(64 * multiplier, 128 * multiplier), ResidualSpatial(128 * multiplier, 192 * multiplier),
            nn.AdaptiveAvgPool2d(2), nn.Flatten(),
        )
        self.temporal = TemporalTokens(768 * multiplier, width=96 * multiplier)

    def forward(self, values):
        batch, steps, channels, height, width = values.shape
        features = self.frames(values.reshape(batch * steps, channels, height, width))
        return self.temporal(features.reshape(batch, steps, -1))


class ScratchMultiModalModel(nn.Module):
    def __init__(self, modalities, fusion="concat", visual_multiplier=1, skeleton_representation="coordinates",
                 thermal_view="full", imu_representation="raw", visual_encoder="legacy", visual_feedforward=384,
                 scratch_skeleton_encoder="tcn", skeleton_dimension=192, skeleton_depth=3, skeleton_heads=6,
                 imu_token_encoder="conv_lstm"):
        super().__init__()
        validate_scratch_config({"modalities": modalities, "fusion": fusion, "visual_multiplier": visual_multiplier,
                                 "skeleton_representation": skeleton_representation,
                                 "thermal_view": thermal_view, "imu_representation": imu_representation,
                                 "visual_encoder": visual_encoder, "visual_feedforward": visual_feedforward,
                                 "scratch_skeleton_encoder": scratch_skeleton_encoder,
                                 "skeleton_dimension": skeleton_dimension, "skeleton_depth": skeleton_depth,
                                 "skeleton_heads": skeleton_heads, "imu_token_encoder": imu_token_encoder})
        from src.spatial_tokens import SpatialVideoTokens
        self.modalities = modalities
        self.fusion = fusion
        self.thermal_view = thermal_view
        self.imu_representation = imu_representation
        self.register_buffer("class_prior_logits", torch.zeros(40))
        imu_encoder = {
            "conv_lstm": ConvLSTMTokens,
            "statistics": StatisticalTokens,
            "transformer": TransformerIMUTokens,
        }[imu_token_encoder]
        self.encoders = nn.ModuleDict({
            name: (DualPathSkeletonTokens(skeleton_dimension, skeleton_depth, skeleton_heads)
                   if scratch_skeleton_encoder == "dstformer" else SkeletonTokens(skeleton_representation))
            if name == "skeleton" else imu_encoder({
                "imu": 85, "imu_world": 85, "imu_relative": 95, "imu_official": 50, "imu_magnetic": 100,
            }[name])
            if name in ("imu", "imu_world", "imu_relative", "imu_official", "imu_magnetic")
            else TemporalTokens(28) if name == "radar" else VideoTokens(visual_multiplier) if visual_encoder == "legacy"
            else SpatialVideoTokens(visual_encoder, visual_feedforward)
            for name in modalities
        })
        self.auxiliary = nn.ModuleDict({name: nn.Linear(192, 40) for name in modalities})
        self.head = nn.Sequential(
            nn.Linear(len(modalities) * 193 if fusion == "concat" else 192, 384),
            nn.LayerNorm(384), nn.SiLU(), nn.Dropout(0.25), nn.Linear(384, 40),
        )
        if fusion in ("transformer", "cross_transformer"):
            self.class_token = nn.Parameter(torch.zeros(1, 1, 192))
            self.time_embedding = nn.Parameter(torch.zeros(1, 16, 192))
            self.modality_embedding = nn.Parameter(torch.zeros(len(modalities), 1, 192))
            nn.init.trunc_normal_(self.time_embedding, std=0.02)
            nn.init.trunc_normal_(self.modality_embedding, std=0.02)
            if fusion == "transformer":
                layer = nn.TransformerEncoderLayer(192, 4, 384, dropout=0.1, batch_first=True, norm_first=True)
                self.transformer = nn.TransformerEncoder(layer, 2, norm=nn.LayerNorm(192), enable_nested_tensor=False)
            else:
                layer = nn.TransformerDecoderLayer(192, 4, 384, dropout=0.1, batch_first=True, norm_first=True)
                self.transformer = nn.TransformerDecoder(layer, 2, norm=nn.LayerNorm(192))

    def forward(self, batch):
        features, tokens, missing = [], [], []
        auxiliary = {}
        for position, name in enumerate(self.modalities):
            mask = batch[name + "_mask"].reshape(-1, 1)
            values = batch[name]
            values = torch.where(mask.reshape(-1, *([1] * (values.ndim - 1))) > 0, values, 0)
            encoded = self.encoders[name](values)
            pooled = encoded.mean(1)
            auxiliary[name] = self.auxiliary[name](pooled)
            features.extend([pooled * mask, mask])
            if self.fusion in ("transformer", "cross_transformer"):
                if encoded.shape[1] % 16:
                    raise ValueError("Encoder tokens must be time-major with 16 time steps")
                time_embedding = self.time_embedding.repeat_interleave(encoded.shape[1] // 16, dim=1)
                tokens.append(encoded + time_embedding + self.modality_embedding[position])
                missing.append((mask <= 0).expand(-1, encoded.shape[1]))
        if self.fusion == "concat":
            output = self.head(torch.cat(features, dim=1))
        else:
            batch_size = tokens[0].shape[0]
            values = torch.cat([self.class_token.expand(batch_size, -1, -1), *tokens], dim=1)
            padding = torch.cat([torch.zeros(batch_size, 1, dtype=torch.bool, device=values.device), *missing], dim=1)
            if self.fusion == "transformer":
                encoded = self.transformer(values, src_key_padding_mask=padding)
                pooled = encoded[:, 0]
            else:
                queries = (self.class_token + self.time_embedding).expand(batch_size, -1, -1)
                encoded = self.transformer(queries, values, memory_key_padding_mask=padding)
                pooled = encoded.mean(1)
            output = self.head(pooled)
        available = torch.stack([batch[name + "_mask"] > 0 for name in self.modalities]).any(0)
        return torch.where(available[:, None], output, self.class_prior_logits), auxiliary

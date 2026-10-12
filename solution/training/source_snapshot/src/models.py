import torch
from torch import nn


class TemporalBlock(nn.Module):
    def __init__(self, channels, dilation=1, dropout=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(channels, channels, 5, padding=2 * dilation, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels), nn.SiLU(), nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.SiLU()

    def forward(self, values):
        return self.activation(values + self.layers(values))


class TemporalEncoder(nn.Module):
    def __init__(self, inputs, width=96, dropout=0.15):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(inputs, width, 1, bias=False), nn.BatchNorm1d(width), nn.SiLU(),
            TemporalBlock(width, 1, dropout), TemporalBlock(width, 2, dropout),
            TemporalBlock(width, 4, dropout),
        )
        self.project = nn.Sequential(nn.Linear(width * 3, 192), nn.LayerNorm(192), nn.SiLU())

    def forward(self, values):
        features = self.layers(values.transpose(1, 2))
        pooled = torch.cat([features.mean(-1), features.amax(-1), features.std(-1, unbiased=False)], dim=1)
        return self.project(pooled)


class SpatialBlock(nn.Module):
    def __init__(self, inputs, outputs, stride):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(inputs, outputs, 3, stride, 1, bias=False), nn.BatchNorm2d(outputs), nn.SiLU(),
            nn.Conv2d(outputs, outputs, 3, 1, 1, groups=outputs, bias=False),
            nn.BatchNorm2d(outputs), nn.SiLU(),
            nn.Conv2d(outputs, outputs, 1, bias=False), nn.BatchNorm2d(outputs), nn.SiLU(),
        )

    def forward(self, values):
        return self.layers(values)


class VideoEncoder(nn.Module):
    def __init__(self, inputs, spatial_pool=1):
        super().__init__()
        self.frames = nn.Sequential(
            SpatialBlock(inputs, 24, 2), SpatialBlock(24, 48, 2),
            SpatialBlock(48, 96, 2), SpatialBlock(96, 160, 2),
            nn.AdaptiveAvgPool2d(spatial_pool), nn.Flatten(),
        )
        self.temporal = TemporalEncoder(160 * spatial_pool ** 2, width=96)

    def forward(self, video):
        batch, steps, channels, height, width = video.shape
        features = self.frames(video.reshape(batch * steps, channels, height, width))
        return self.temporal(features.reshape(batch, steps, -1))


class MobileVideoEncoder(nn.Module):
    def __init__(self, inputs, spatial_pool=1, imagenet=False, backbone_name="mobile_small", temporal_shift=False):
        super().__init__()
        self.temporal_shift = temporal_shift
        from torchvision.models import EfficientNet_B0_Weights, MobileNet_V3_Small_Weights, efficientnet_b0, mobilenet_v3_small
        if backbone_name == "efficientnet_b0":
            backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if imagenet else None)
            feature_channels = 1280
        else:
            backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if imagenet else None)
            feature_channels = 576
        if inputs != 3:
            previous = backbone.features[0][0]
            first = nn.Conv2d(inputs, previous.out_channels, 3, 2, 1, bias=False)
            with torch.no_grad():
                first.weight[:, :3].copy_(previous.weight * 0.75)
                first.weight[:, 3:].copy_(previous.weight.mean(1, keepdim=True) * 0.75)
            backbone.features[0][0] = first
        self.frames = nn.Sequential(backbone.features, nn.AdaptiveAvgPool2d(spatial_pool), nn.Flatten())
        self.temporal = TemporalEncoder(feature_channels * spatial_pool ** 2, width=128)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406, 0.449][:inputs]).reshape(1, inputs, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225, 0.226][:inputs]).reshape(1, inputs, 1, 1))

    def forward(self, video):
        batch, steps, channels, height, width = video.shape
        images = video.reshape(batch * steps, channels, height, width)
        images = ((images + 1) * 0.5 - self.mean) / self.std
        if self.temporal_shift:
            features = images
            for layer in self.frames[0]:
                if getattr(layer, "use_res_connect", False):
                    shortcut = features
                    features = layer.block(self.shift(features, steps))
                    if hasattr(layer, "stochastic_depth"):
                        features = layer.stochastic_depth(features)
                    features = features + shortcut
                else:
                    features = layer(features)
            features = self.frames[1:](features)
        else:
            features = self.frames(images)
        return self.temporal(features.reshape(batch, steps, -1))

    @staticmethod
    def shift(features, steps):
        batch_steps, channels, height, width = features.shape
        video = features.reshape(batch_steps // steps, steps, channels, height, width)
        shifted = torch.zeros_like(video)
        boundary = channels // 8
        shifted[:, :-1, :boundary] = video[:, 1:, :boundary]
        shifted[:, 1:, boundary:2 * boundary] = video[:, :-1, boundary:2 * boundary]
        shifted[:, :, 2 * boundary:] = video[:, :, 2 * boundary:]
        return shifted.reshape_as(features)


class ResNetVideoEncoder(nn.Module):
    def __init__(self, imagenet=False, dimension=256, maximum_frames=32):
        super().__init__()
        from torchvision.models import ResNet50_Weights, resnet50

        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if imagenet else None)
        self.frames = nn.Sequential(*list(backbone.children())[:-2], nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.projection = nn.Sequential(
            nn.Linear(2048, dimension), nn.LayerNorm(dimension), nn.GELU()
        )
        layer = nn.TransformerEncoderLayer(
            dimension,
            nhead=8,
            dim_feedforward=dimension * 3,
            dropout=0.15,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.class_token = nn.Parameter(torch.zeros(1, 1, dimension))
        self.position_embedding = nn.Parameter(torch.zeros(1, maximum_frames + 1, dimension))
        self.output = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, 192), nn.SiLU())
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, video):
        batch, steps, channels, height, width = video.shape
        if channels != 3 or steps + 1 > self.position_embedding.shape[1]:
            raise ValueError("ResNet video input must be three-channel and within the frame budget")
        images = video.reshape(batch * steps, channels, height, width)
        images = ((images + 1) * 0.5 - self.mean) / self.std
        features = self.frames(images).reshape(batch, steps, -1)
        features = self.projection(features)
        token = self.class_token.expand(batch, -1, -1)
        tokens = torch.cat([token, features], dim=1) + self.position_embedding[:, :steps + 1]
        return self.output(self.temporal(tokens)[:, 0])


class X3DVideoEncoder(nn.Module):
    def __init__(self, kinetics=False, kinetics_source="s"):
        super().__init__()
        import sys
        from pathlib import Path
        dependencies = str(Path(__file__).resolve().parents[1] / ".tools/python")
        if dependencies not in sys.path:
            sys.path.insert(0, dependencies)
        from pytorchvideo.models.hub.x3d import x3d_l, x3d_m, x3d_s
        builders = {"s": x3d_s, "m": x3d_m, "l": x3d_l}
        if kinetics_source not in builders:
            raise ValueError(f"Unknown X3D source: {kinetics_source}")
        backbone = builders[kinetics_source](pretrained=kinetics)
        self.target_size = {"s": 160, "m": 224, "l": 312}[kinetics_source]
        head_pool = backbone.blocks[-1].pool
        head_pool.pool = nn.AdaptiveAvgPool3d(1)
        self.frames = nn.Sequential(*backbone.blocks[:-1], head_pool, nn.Flatten())
        self.project = nn.Sequential(nn.Dropout(0.3), nn.Linear(2048, 192), nn.LayerNorm(192), nn.SiLU())

    def forward(self, video):
        batch, steps, channels, height, width = video.shape
        images = torch.nn.functional.interpolate(video.reshape(batch * steps, channels, height, width),
                                                 size=(self.target_size, self.target_size), mode="bilinear", align_corners=False)
        images = ((images + 1) * 0.5 - 0.45) / 0.225
        features = self.frames(
            images.reshape(batch, steps, channels, self.target_size, self.target_size).transpose(1, 2)
        )
        return self.project(features)


class Swin3DVideoEncoder(nn.Module):
    def __init__(self, kinetics=False):
        super().__init__()
        from torchvision.models.video import Swin3D_T_Weights, swin3d_t

        backbone = swin3d_t(weights=Swin3D_T_Weights.KINETICS400_V1 if kinetics else None)
        self.frames = nn.Sequential(
            backbone.patch_embed,
            backbone.pos_drop,
            backbone.features,
            backbone.norm,
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.project = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(768, 192), nn.LayerNorm(192), nn.SiLU()
        )
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, 1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 1, 3, 1, 1))

    def forward(self, video):
        batch, steps, channels, height, width = video.shape
        if channels != 3:
            raise ValueError("Swin3D requires three-channel video")
        video = ((video + 1) * 0.5 - self.mean) / self.std
        features = self.frames(video.permute(0, 2, 1, 3, 4)).permute(0, 4, 1, 2, 3)
        return self.project(self.pool(features).flatten(1))


class MViTVideoEncoder(nn.Module):
    def __init__(self, kinetics=False, dimension=192):
        super().__init__()
        from torchvision.models.video import MViT_V2_S_Weights, mvit_v2_s

        backbone = mvit_v2_s(
            weights=MViT_V2_S_Weights.KINETICS400_V1 if kinetics else None
        )
        feature_dimension = backbone.head[-1].in_features
        backbone.head = nn.Identity()
        self.frames = backbone
        self.project = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dimension, dimension),
            nn.LayerNorm(dimension),
            nn.GELU(),
        )
        self.register_buffer(
            "mean", torch.tensor([0.45, 0.45, 0.45]).reshape(1, 3, 1, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.225, 0.225, 0.225]).reshape(1, 3, 1, 1, 1)
        )

    def forward(self, video):
        _, _, channels, _, _ = video.shape
        if channels != 3:
            raise ValueError("MViT requires three-channel video")
        video = video.permute(0, 2, 1, 3, 4)
        if video.shape[2:] != (16, 224, 224):
            video = torch.nn.functional.interpolate(
                video,
                size=(16, 224, 224),
                mode="trilinear",
                align_corners=False,
            )
        video = ((video + 1) * 0.5 - self.mean) / self.std
        return self.project(self.frames(video))


class SharedMViTModel(nn.Module):
    visual_modalities = {
        "thermal",
        "scene_ir",
        "scene_depth",
        "person_ir",
        "person_thermal",
        "person_depth",
        "person_ir_detail",
        "person_thermal_detail",
        "yolo_rgb",
        "yolo_ir",
    }

    def __init__(self, modalities, kinetics=False, dimension=192, visual_encoder=None):
        super().__init__()
        if not modalities or len(set(modalities)) != len(modalities):
            raise ValueError("Shared MViT modalities must be non-empty and unique")
        sensor_channels = {
            "skeleton": 153,
            "imu": 85,
            "imu_world": 85,
            "imu_timed": 105,
            "imu_relative": 95,
            "imu_official": 50,
            "imu_magnetic": 100,
            "radar": 28,
            "geometry": 408,
        }
        unsupported = set(modalities) - self.visual_modalities - set(sensor_channels)
        if unsupported:
            raise ValueError(f"Unsupported shared MViT modalities: {sorted(unsupported)}")
        if not set(modalities) & self.visual_modalities:
            raise ValueError("Shared MViT fusion requires at least one visual modality")
        self.modalities = modalities
        self.visual = visual_encoder or MViTVideoEncoder(kinetics, dimension)
        self.sensor_encoders = nn.ModuleDict(
            {
                name: TemporalEncoder(sensor_channels[name], width=128)
                for name in modalities
                if name in sensor_channels
            }
        )
        self.class_token = nn.Parameter(torch.zeros(1, 1, dimension))
        self.modality_embedding = nn.Parameter(
            torch.zeros(1, len(modalities), dimension)
        )
        layer = nn.TransformerEncoderLayer(
            dimension,
            nhead=6,
            dim_feedforward=dimension * 3,
            dropout=0.15,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(
            layer,
            num_layers=2,
            norm=nn.LayerNorm(dimension),
            enable_nested_tensor=False,
        )
        self.auxiliary = nn.ModuleDict(
            {name: nn.Linear(dimension, 40) for name in modalities}
        )
        self.expert_gate = nn.Linear(dimension, len(modalities) * 40)
        self.residual_head = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(dimension, 40),
        )
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.modality_embedding, std=0.02)
        nn.init.zeros_(self.expert_gate.weight)
        nn.init.zeros_(self.expert_gate.bias)
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def forward(self, batch, return_embedding=False):
        features = {}
        visual_names = [
            name for name in self.modalities if name in self.visual_modalities
        ]
        visual_values = []
        for name in visual_names:
            values = batch[name]
            mask = batch[name + "_mask"].reshape(-1, 1, 1, 1, 1)
            visual_values.append(torch.where(mask > 0, values, 0))
        visual_features = self.visual(torch.cat(visual_values, dim=0))
        for name, encoded in zip(visual_names, visual_features.chunk(len(visual_names))):
            features[name] = encoded
        for name, encoder in self.sensor_encoders.items():
            mask = batch[name + "_mask"].reshape(-1, 1, 1)
            features[name] = encoder(torch.where(mask > 0, batch[name], 0))

        tokens = torch.stack([features[name] for name in self.modalities], dim=1)
        tokens = tokens + self.modality_embedding
        batch_size = tokens.shape[0]
        class_token = self.class_token.expand(batch_size, -1, -1)
        padding = torch.stack(
            [batch[name + "_mask"] <= 0 for name in self.modalities], dim=1
        )
        padding = torch.cat(
            [
                torch.zeros(batch_size, 1, dtype=torch.bool, device=tokens.device),
                padding,
            ],
            dim=1,
        )
        embedding = self.fusion(
            torch.cat([class_token, tokens], dim=1), src_key_padding_mask=padding
        )[:, 0]
        auxiliary = {
            name: self.auxiliary[name](features[name]) for name in self.modalities
        }
        expert_logits = torch.stack(
            [auxiliary[name] for name in self.modalities], dim=1
        )
        gates = self.expert_gate(embedding).reshape(
            batch_size, len(self.modalities), 40
        )
        availability = torch.stack(
            [batch[name + "_mask"] > 0 for name in self.modalities], dim=1
        )
        gates = gates.masked_fill(~availability.unsqueeze(-1), -1e4).softmax(dim=1)
        output = (gates * expert_logits).sum(dim=1) + self.residual_head(embedding)
        if return_embedding:
            return output, auxiliary, embedding
        return output, auxiliary


class SharedSwin3DModel(SharedMViTModel):
    def __init__(self, modalities, kinetics=False, dimension=192):
        super().__init__(
            modalities,
            kinetics=False,
            dimension=dimension,
            visual_encoder=Swin3DVideoEncoder(kinetics),
        )


class X3DTokenEncoder(nn.Module):
    def __init__(self, kinetics=False, kinetics_source="s", dimension=192):
        super().__init__()
        import sys
        from pathlib import Path
        dependencies = str(Path(__file__).resolve().parents[1] / ".tools/python")
        if dependencies not in sys.path:
            sys.path.insert(0, dependencies)
        from pytorchvideo.models.hub.x3d import x3d_l, x3d_m, x3d_s
        builders = {"s": x3d_s, "m": x3d_m, "l": x3d_l}
        if kinetics_source not in builders:
            raise ValueError(f"Unknown X3D source: {kinetics_source}")
        backbone = builders[kinetics_source](pretrained=kinetics)
        self.target_size = {"s": 160, "m": 224, "l": 312}[kinetics_source]
        self.frames = nn.Sequential(*backbone.blocks[:-1])
        channels = {"s": 192, "m": 192, "l": 192}[kinetics_source]
        self.project = nn.Sequential(
            nn.LayerNorm(channels), nn.Linear(channels, dimension), nn.GELU()
        )

    def forward(self, video):
        batch, steps, channels, height, width = video.shape
        if channels != 3:
            raise ValueError("Shared X3D token inputs must be three-channel videos")
        images = torch.nn.functional.interpolate(
            video.reshape(batch * steps, channels, height, width),
            size=(self.target_size, self.target_size),
            mode="bilinear",
            align_corners=False,
        )
        images = ((images + 1) * 0.5 - 0.45) / 0.225
        feature_map = self.frames(
            images.reshape(batch, steps, channels, self.target_size, self.target_size).transpose(1, 2)
        )
        tokens = feature_map.mean(dim=(-2, -1)).transpose(1, 2)
        return self.project(tokens)


class SharedX3DTokenModel(nn.Module):
    visual_modalities = {
        "thermal", "scene_ir", "scene_depth", "person_ir", "person_thermal",
        "person_depth", "person_ir_detail", "person_thermal_detail", "yolo_rgb", "yolo_ir",
    }

    visual_groups = {
        "person_depth": "rgb",
        "person_ir": "infrared",
        "person_ir_detail": "infrared",
        "scene_ir": "infrared",
        "person_thermal": "thermal",
        "person_thermal_detail": "thermal",
        "thermal": "thermal",
        "scene_depth": "depth",
        "yolo_rgb": "rgb",
        "yolo_ir": "infrared",
    }

    def __init__(self, modalities, kinetics=False, kinetics_source="s", dimension=192, grouped_visual=False):
        super().__init__()
        if not modalities or len(set(modalities)) != len(modalities):
            raise ValueError("Shared X3D modalities must be non-empty and unique")
        unsupported = set(modalities) - self.visual_modalities - {"skeleton", "imu", "imu_world", "radar"}
        if unsupported:
            raise ValueError(f"Unsupported shared X3D modalities: {sorted(unsupported)}")
        if not set(modalities) & self.visual_modalities:
            raise ValueError("Shared X3D token fusion requires at least one visual modality")
        from src.scratch_models import ConvLSTMTokens, SkeletonTokens, TemporalTokens

        self.modalities = modalities
        self.grouped_visual = grouped_visual
        if grouped_visual:
            groups = sorted({self.visual_groups[name] for name in modalities if name in self.visual_modalities})
            self.visual_encoders = nn.ModuleDict(
                {name: X3DTokenEncoder(kinetics, kinetics_source, dimension) for name in groups}
            )
        else:
            self.visual = X3DTokenEncoder(kinetics, kinetics_source, dimension)
        self.sensor_encoders = nn.ModuleDict()
        for name in modalities:
            if name == "skeleton":
                self.sensor_encoders[name] = SkeletonTokens("coordinates")
            elif name in ("imu", "imu_world"):
                self.sensor_encoders[name] = ConvLSTMTokens(85)
            elif name == "radar":
                self.sensor_encoders[name] = TemporalTokens(28)
        self.class_token = nn.Parameter(torch.zeros(1, 1, dimension))
        self.time_embedding = nn.Parameter(torch.zeros(1, 16, dimension))
        self.modality_embedding = nn.Parameter(torch.zeros(len(modalities), 1, dimension))
        layer = nn.TransformerEncoderLayer(
            dimension,
            nhead=6,
            dim_feedforward=dimension * 4,
            dropout=0.15,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(
            layer, num_layers=3, norm=nn.LayerNorm(dimension), enable_nested_tensor=False
        )
        self.auxiliary = nn.ModuleDict({name: nn.Linear(dimension, 40) for name in modalities})
        self.expert_gate = nn.Linear(dimension, len(modalities) * 40)
        self.residual_head = nn.Sequential(
            nn.Linear(dimension, dimension), nn.GELU(), nn.Dropout(0.25), nn.Linear(dimension, 40)
        )
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.time_embedding, std=0.02)
        nn.init.trunc_normal_(self.modality_embedding, std=0.02)
        nn.init.zeros_(self.expert_gate.weight)
        nn.init.zeros_(self.expert_gate.bias)
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def forward(self, batch, return_embedding=False):
        streams, missing, expert_logits, auxiliary = [], [], [], {}
        for position, name in enumerate(self.modalities):
            mask = batch[name + "_mask"].reshape(-1, 1)
            values = batch[name]
            values = torch.where(
                mask.reshape(-1, *([1] * (values.ndim - 1))) > 0,
                values,
                0,
            )
            if name in self.visual_modalities:
                encoder = self.visual_encoders[self.visual_groups[name]] if self.grouped_visual else self.visual
                encoded = encoder(values)
            else:
                encoded = self.sensor_encoders[name](values)
            if encoded.shape[1] != 16:
                encoded = torch.nn.functional.adaptive_avg_pool1d(
                    encoded.transpose(1, 2), 16
                ).transpose(1, 2)
            encoded = encoded + self.time_embedding + self.modality_embedding[position]
            streams.append(encoded)
            missing.append((mask <= 0).expand(-1, encoded.shape[1]))
            logits = self.auxiliary[name](encoded.mean(1))
            auxiliary[name] = logits
            expert_logits.append(logits)
        batch_size = streams[0].shape[0]
        tokens = torch.cat([self.class_token.expand(batch_size, -1, -1), *streams], dim=1)
        padding = torch.cat(
            [torch.zeros(batch_size, 1, dtype=torch.bool, device=tokens.device), *missing], dim=1
        )
        embedding = self.fusion(tokens, src_key_padding_mask=padding)[:, 0]
        gates = self.expert_gate(embedding).reshape(batch_size, len(self.modalities), 40)
        availability = torch.stack([batch[name + "_mask"] > 0 for name in self.modalities], dim=1)
        gates = gates.masked_fill(~availability.unsqueeze(-1), -1e4).softmax(dim=1)
        output = (gates * torch.stack(expert_logits, dim=1)).sum(dim=1) + self.residual_head(embedding)
        if return_embedding:
            return output, auxiliary, embedding
        return output, auxiliary


class GroupedX3DTokenModel(SharedX3DTokenModel):
    def __init__(self, modalities, kinetics=False, kinetics_source="s", dimension=192):
        super().__init__(modalities, kinetics, kinetics_source, dimension, grouped_visual=True)


class GraphTemporalBlock(nn.Module):
    def __init__(self, inputs, outputs, stride=1):
        super().__init__()
        adjacency = torch.zeros(3, 17, 17)
        adjacency[0] = torch.eye(17)
        parents = [0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
        for joint, parent in enumerate(parents):
            if joint != parent:
                adjacency[1, joint, parent] = 1
                adjacency[2, parent, joint] = 1
        adjacency /= adjacency.sum(-2, keepdim=True).clamp_min(1)
        self.register_buffer("adjacency", adjacency)
        self.adaptation = nn.Parameter(torch.zeros_like(adjacency))
        self.spatial = nn.Conv2d(inputs, outputs * 3, 1)
        self.temporal = nn.Sequential(
            nn.BatchNorm2d(outputs), nn.SiLU(),
            nn.Conv2d(outputs, outputs, (9, 1), (stride, 1), (4, 0), bias=False),
            nn.BatchNorm2d(outputs), nn.Dropout(0.15),
        )
        self.residual = nn.Identity() if inputs == outputs and stride == 1 else nn.Sequential(
            nn.Conv2d(inputs, outputs, 1, (stride, 1), bias=False), nn.BatchNorm2d(outputs))
        self.activation = nn.SiLU()

    def forward(self, values):
        spatial = self.spatial(values)
        batch, channels, steps, joints = spatial.shape
        spatial = spatial.reshape(batch, 3, channels // 3, steps, joints)
        spatial = torch.einsum("bkctv,kvw->bctw", spatial, self.adjacency + self.adaptation)
        return self.activation(self.temporal(spatial) + self.residual(values))


class SkeletonGraphEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.normalization = nn.BatchNorm1d(153)
        self.layers = nn.Sequential(
            GraphTemporalBlock(9, 48), GraphTemporalBlock(48, 48),
            GraphTemporalBlock(48, 96, 2), GraphTemporalBlock(96, 96),
            GraphTemporalBlock(96, 192, 2), GraphTemporalBlock(192, 192),
        )
        self.project = nn.Sequential(nn.Linear(384, 192), nn.LayerNorm(192), nn.SiLU())

    def forward(self, values):
        batch, steps, _ = values.shape
        values = self.normalization(values.transpose(1, 2)).transpose(1, 2)
        values = values.reshape(batch, steps, 3, 17, 3).permute(0, 2, 4, 1, 3).reshape(batch, 9, steps, 17)
        features = self.layers(values)
        return self.project(torch.cat([features.mean((2, 3)), features.amax((2, 3))], dim=1))


class JointAttentionBlock(nn.Module):
    def __init__(self, width=96, stride=1):
        super().__init__()
        self.normalization = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, 4, dropout=0.1, batch_first=True)
        self.temporal = nn.Sequential(
            nn.Conv2d(width, width, (7, 1), padding=(3, 0), groups=width, bias=False),
            nn.BatchNorm2d(width), nn.SiLU(), nn.Conv2d(width, width, 1, bias=False),
            nn.BatchNorm2d(width), nn.Dropout(0.15),
        )
        self.activation = nn.SiLU()
        self.pool = nn.AvgPool2d((stride, 1)) if stride > 1 else nn.Identity()

    def forward(self, values):
        batch, channels, steps, joints = values.shape
        tokens = values.permute(0, 2, 3, 1).reshape(batch * steps, joints, channels)
        normalized = self.normalization(tokens)
        attended = self.attention(normalized, normalized, normalized, need_weights=False)[0]
        values = (tokens + attended).reshape(batch, steps, joints, channels).permute(0, 3, 1, 2)
        return self.pool(self.activation(values + self.temporal(values)))


class JointAttentionEncoder(nn.Module):
    def __init__(self, width=96):
        super().__init__()
        self.normalization = nn.BatchNorm1d(153)
        self.embedding = nn.Parameter(torch.zeros(1, width, 1, 17))
        nn.init.trunc_normal_(self.embedding, std=0.02)
        self.input = nn.Conv2d(9, width, 1)
        self.layers = nn.Sequential(JointAttentionBlock(width), JointAttentionBlock(width, 2),
                                    JointAttentionBlock(width, 2))
        self.project = nn.Sequential(nn.Dropout(0.25), nn.Linear(width * 17 * 2, 192),
                                     nn.LayerNorm(192), nn.SiLU())

    def forward(self, values):
        batch, steps, _ = values.shape
        values = self.normalization(values.transpose(1, 2)).transpose(1, 2)
        values = values.reshape(batch, steps, 3, 17, 3).permute(0, 2, 4, 1, 3).reshape(batch, 9, steps, 17)
        features = self.layers(self.input(values) + self.embedding)
        pooled = torch.cat([features.mean(2).flatten(1), features.amax(2).flatten(1)], dim=1)
        return self.project(pooled)


class StatisticalEncoder(nn.Module):
    def __init__(self, inputs, pyramid=False):
        super().__init__()
        self.pyramid = pyramid
        pooled_inputs = inputs * (18 if pyramid else 6)
        self.project = nn.Sequential(
            nn.BatchNorm1d(pooled_inputs), nn.Dropout(0.25), nn.Linear(pooled_inputs, 256),
            nn.LayerNorm(256), nn.SiLU(), nn.Dropout(0.35), nn.Linear(256, 192), nn.SiLU(),
        )

    def forward(self, values):
        delta = torch.diff(values, dim=1)
        pooled = torch.cat([values.mean(1), values.std(1, unbiased=False), values.amax(1), values.amin(1),
                            delta.abs().mean(1), delta.std(1, unbiased=False)], dim=1)
        if self.pyramid:
            parts = [pooled]
            for segments in [2, 4]:
                for segment in torch.tensor_split(values, segments, dim=1):
                    parts.extend([segment.mean(1), segment.std(1, unbiased=False)])
            pooled = torch.cat(parts, dim=1)
        return self.project(pooled)


class MultiModalModel(nn.Module):
    def __init__(self, modalities, width=96, dropout=0.25, spatial_pool=1, backbone="scratch", imagenet=False,
                 skeleton_backbone="tcn", temporal_backbone="tcn", temporal_shift=False, kinetics=False,
                 kinetics_source="s", fusion="concat"):
        super().__init__()
        if backbone in ("x3d_shared_tokens", "x3d_grouped_tokens"):
            raise ValueError("Use SharedX3DTokenModel for the shared temporal-token backbone")
        if fusion not in ("concat", "transformer", "residual_transformer"):
            raise ValueError(f"Unknown legacy-model fusion: {fusion}")
        self.fusion_mode = fusion
        if backbone == "x3d_s" and "depth_ir" in modalities:
            raise ValueError("X3D requires separate three-channel visual modalities")
        self.modalities = modalities
        channels = {"skeleton": 153, "imu": 85, "imu_world": 85, "imu_timed": 105, "imu_relative": 95,
                    "imu_official": 50, "imu_magnetic": 100, "radar": 28, "geometry": 408,
                    "ir_features": 2560, "thermal_features": 2560, "depth_features": 2560}
        self.encoders = nn.ModuleDict({
            name: TemporalEncoder(channels[name], width) if name in channels
            else (X3DVideoEncoder(kinetics, kinetics_source) if backbone == "x3d_s" else
                  Swin3DVideoEncoder(kinetics) if backbone == "swin3d_t" else
                  ResNetVideoEncoder(imagenet) if backbone == "resnet50" else
                  MobileVideoEncoder(4 if name == "depth_ir" else 3, spatial_pool, imagenet, backbone, temporal_shift)
                  if backbone != "scratch" else VideoEncoder(4 if name == "depth_ir" else 3, spatial_pool))
            for name in modalities
        })
        if "skeleton" in modalities and skeleton_backbone == "stgcn":
            self.encoders["skeleton"] = SkeletonGraphEncoder()
        if "skeleton" in modalities and skeleton_backbone == "attention":
            self.encoders["skeleton"] = JointAttentionEncoder()
        if temporal_backbone in ["statistics", "pyramid"]:
            for name in modalities:
                if name in channels and not (name == "skeleton" and skeleton_backbone == "attention"):
                    self.encoders[name] = StatisticalEncoder(channels[name], pyramid=temporal_backbone == "pyramid")
        self.auxiliary = nn.ModuleDict({name: nn.Linear(192, 40) for name in modalities})
        if fusion in ("concat", "residual_transformer"):
            self.head = nn.Sequential(
                nn.Linear(len(modalities) * 193, 384), nn.LayerNorm(384), nn.SiLU(),
                nn.Dropout(dropout), nn.Linear(384, 40),
            )
        if fusion in ("transformer", "residual_transformer"):
            self.class_token = nn.Parameter(torch.zeros(1, 1, 192))
            self.modality_embedding = nn.Parameter(torch.zeros(1, len(modalities), 192))
            layer = nn.TransformerEncoderLayer(
                192, nhead=6, dim_feedforward=576, dropout=0.15,
                activation="gelu", batch_first=True, norm_first=True,
            )
            self.fusion = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
            self.fusion_normalization = nn.LayerNorm(192)
            if fusion == "transformer":
                self.head = nn.Sequential(
                    nn.Linear(192, 384), nn.LayerNorm(384), nn.SiLU(),
                    nn.Dropout(dropout), nn.Linear(384, 40),
                )
            else:
                self.residual_head = nn.Sequential(
                    nn.Linear(192, 192), nn.GELU(), nn.Dropout(dropout), nn.Linear(192, 40)
                )
                nn.init.zeros_(self.residual_head[-1].weight)
                nn.init.zeros_(self.residual_head[-1].bias)
            nn.init.trunc_normal_(self.class_token, std=0.02)
            nn.init.trunc_normal_(self.modality_embedding, std=0.02)

    def forward(self, batch, return_embedding=False):
        features, masks = [], []
        auxiliary = {}
        for name in self.modalities:
            encoded = self.encoders[name](batch[name])
            auxiliary[name] = self.auxiliary[name](encoded)
            mask = batch[name + "_mask"].reshape(-1, 1)
            features.append(encoded * mask)
            masks.append(mask)
        values = [value for pair in zip(features, masks) for value in pair]
        concatenated = torch.cat(values, dim=1)
        if self.fusion_mode == "concat":
            fused = concatenated
        else:
            tokens = torch.stack(features, dim=1) + self.modality_embedding
            class_token = self.class_token.expand(tokens.shape[0], -1, -1)
            padding_mask = torch.cat(
                [torch.zeros(tokens.shape[0], 1, dtype=torch.bool, device=tokens.device),
                 ~torch.cat(masks, dim=1).bool()],
                dim=1,
            )
            tokens = self.fusion(torch.cat([class_token, tokens], dim=1), src_key_padding_mask=padding_mask)
            fused = self.fusion_normalization(tokens[:, 0])
        if self.fusion_mode == "residual_transformer":
            embedding = self.head[:-1](concatenated)
            output = self.head[-1](embedding) + self.residual_head(fused)
            if return_embedding:
                return output, auxiliary, embedding
            return output, auxiliary
        embedding = self.head[:-1](fused)
        output = self.head[-1](embedding)
        if return_embedding:
            return output, auxiliary, embedding
        return output, auxiliary

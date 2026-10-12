#!/usr/bin/env python3
import copy
from pathlib import Path
import sys

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.v24_iterative_loop.iterative_model import build_base_model, checkpoint_state


class SpatiotemporalResidualAdapter(nn.Module):
    def __init__(self, channels, bottleneck):
        super().__init__()
        groups = min(8, bottleneck)
        while bottleneck % groups:
            groups -= 1
        self.down = nn.Conv3d(channels, bottleneck, kernel_size=1, bias=False)
        self.normalization = nn.GroupNorm(groups, bottleneck)
        self.activation = nn.GELU()
        self.context = nn.Conv3d(
            bottleneck,
            bottleneck,
            kernel_size=3,
            padding=1,
            groups=bottleneck,
            bias=False,
        )
        self.context_normalization = nn.GroupNorm(groups, bottleneck)
        self.up = nn.Conv3d(bottleneck, channels, kernel_size=1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, values):
        residual = self.activation(self.normalization(self.down(values)))
        residual = self.activation(self.context_normalization(self.context(residual)))
        return values + self.up(residual)


class ProtectedAdapterVideoEncoder(nn.Module):
    def __init__(self, source, bottleneck=128, output_adapters=True):
        super().__init__()
        if source.modality_specific_bn or source.separate_stems or source.modality_film:
            raise ValueError("V97 requires the plain joint visual encoder")
        self.stem = source.stem
        self.layer1 = source.layer1
        self.layer2 = source.layer2
        self.layer3 = source.layer3
        self.layer4 = source.layer4
        self.intermediate_adapters = nn.ModuleList(
            [SpatiotemporalResidualAdapter(256, bottleneck) for _ in range(3)]
        )
        self.output_adapters = nn.ModuleList(
            [SpatiotemporalResidualAdapter(512, bottleneck) for _ in range(3)]
            if output_adapters
            else []
        )

    def forward_joint(self, videos, return_multiscale=False):
        if len(videos) != 3:
            raise ValueError("V97 requires exactly three visual modalities")
        values = self.stem(torch.cat(videos, dim=0))
        values = self.layer2(self.layer1(values))
        intermediate = self.layer3(values)
        intermediate_streams = [
            adapter(stream)
            for adapter, stream in zip(self.intermediate_adapters, intermediate.chunk(3, dim=0))
        ]
        output = self.layer4(torch.cat(intermediate_streams, dim=0))
        output_streams = list(output.chunk(3, dim=0))
        if self.output_adapters:
            output_streams = [
                adapter(stream)
                for adapter, stream in zip(self.output_adapters, output_streams)
            ]
        if return_multiscale:
            return list(zip(intermediate_streams, output_streams))
        return output_streams

    def forward(self, video, modality=0, return_multiscale=False):
        if not 0 <= modality < 3:
            raise ValueError("Visual modality index must be in [0, 2]")
        values = self.layer2(self.layer1(self.stem(video)))
        intermediate = self.intermediate_adapters[modality](self.layer3(values))
        output = self.layer4(intermediate)
        if self.output_adapters:
            output = self.output_adapters[modality](output)
        return (intermediate, output) if return_multiscale else output

    def adapter_modules(self):
        return [*self.intermediate_adapters, *self.output_adapters]


class ProtectedVisualAdapterClassifier(nn.Module):
    def __init__(self, base, bottleneck=128, output_adapters=True):
        super().__init__()
        base.video = ProtectedAdapterVideoEncoder(
            base.video,
            bottleneck=bottleneck,
            output_adapters=output_adapters,
        )
        base.per_visual_auxiliary = True
        self.base = base
        self.bottleneck = bottleneck
        self.use_output_adapters = output_adapters
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        for module in self.base.video.adapter_modules():
            for parameter in module.parameters():
                parameter.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        for module in self.base.video.adapter_modules():
            module.train(mode)
        return self

    def forward(self, video, skeleton=None, imu=None, radar=None, sensor_mask=None, visual_mask=None):
        return self.base(
            video,
            skeleton=skeleton,
            imu=imu,
            radar=radar,
            sensor_mask=sensor_mask,
            visual_mask=visual_mask,
        )

    def trainable_parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def build_from_source(source_path, bottleneck=128, output_adapters=True):
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if config["mode"] != "shared_multimodal" or config.get("use_radar", False):
        raise ValueError("V97 requires a shared no-radar source")
    if config.get("visual_strategy") != "joint":
        raise ValueError("V97 requires joint visual encoding")
    base = build_base_model(config)
    base.load_state_dict(checkpoint_state(checkpoint), strict=True)
    teacher = copy.deepcopy(base).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    model = ProtectedVisualAdapterClassifier(
        base,
        bottleneck=bottleneck,
        output_adapters=output_adapters,
    )
    return model, teacher, config

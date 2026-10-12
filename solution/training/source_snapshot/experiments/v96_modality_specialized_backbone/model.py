#!/usr/bin/env python3
import copy
from pathlib import Path
import sys

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.v24_iterative_loop.iterative_model import build_base_model, checkpoint_state


class ModalitySpecializedVideoEncoder(nn.Module):
    def __init__(self, source, split_stage=4):
        super().__init__()
        if split_stage not in (3, 4):
            raise ValueError("Visual specialization must start at layer3 or layer4")
        if source.modality_specific_bn or source.separate_stems or source.modality_film:
            raise ValueError("V96 requires the plain joint visual encoder")
        self.split_stage = split_stage
        self.stem = source.stem
        self.layer1 = source.layer1
        self.layer2 = source.layer2
        self.layer3 = source.layer3
        self.layer4 = source.layer4
        self.extra_layer3 = nn.ModuleList(
            [copy.deepcopy(source.layer3) for _ in range(2)] if split_stage == 3 else []
        )
        self.extra_layer4 = nn.ModuleList([copy.deepcopy(source.layer4) for _ in range(2)])

    def forward_joint(self, videos, return_multiscale=False):
        if len(videos) != 3:
            raise ValueError("V96 requires exactly three visual modalities")
        values = self.stem(torch.cat(videos, dim=0))
        values = self.layer1(values)
        values = self.layer2(values)
        if self.split_stage == 4:
            intermediate = self.layer3(values)
            intermediate_streams = intermediate.chunk(3, dim=0)
        else:
            low_streams = values.chunk(3, dim=0)
            layer3_modules = [self.layer3, *self.extra_layer3]
            intermediate_streams = tuple(
                layer(stream) for layer, stream in zip(layer3_modules, low_streams)
            )
        layer4_modules = [self.layer4, *self.extra_layer4]
        output_streams = [
            layer(stream) for layer, stream in zip(layer4_modules, intermediate_streams)
        ]
        if return_multiscale:
            return list(zip(intermediate_streams, output_streams))
        return output_streams

    def forward(self, video, modality=0, return_multiscale=False):
        if not 0 <= modality < 3:
            raise ValueError("Visual modality index must be in [0, 2]")
        values = self.layer2(self.layer1(self.stem(video)))
        layer3 = self.layer3 if modality == 0 or self.split_stage == 4 else self.extra_layer3[modality - 1]
        intermediate = layer3(values)
        layer4 = self.layer4 if modality == 0 else self.extra_layer4[modality - 1]
        output = layer4(intermediate)
        return (intermediate, output) if return_multiscale else output

    def specialized_modules(self):
        modules = [self.layer4, *self.extra_layer4]
        if self.split_stage == 3:
            modules = [self.layer3, *self.extra_layer3, *modules]
        return modules


class ModalitySpecializedClassifier(nn.Module):
    def __init__(self, base, split_stage=4):
        super().__init__()
        base.video = ModalitySpecializedVideoEncoder(base.video, split_stage=split_stage)
        base.per_visual_auxiliary = True
        self.base = base
        self.split_stage = split_stage
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        for module in self.base.video.specialized_modules():
            for parameter in module.parameters():
                parameter.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        for module in self.base.video.specialized_modules():
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


def build_from_source(source_path, split_stage=4):
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if config["mode"] != "shared_multimodal" or config.get("use_radar", False):
        raise ValueError("V96 requires a shared no-radar source")
    if config.get("visual_strategy") != "joint":
        raise ValueError("V96 requires joint visual encoding")
    base = build_base_model(config)
    base.load_state_dict(checkpoint_state(checkpoint), strict=True)
    teacher = copy.deepcopy(base).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return ModalitySpecializedClassifier(base, split_stage=split_stage), teacher, config

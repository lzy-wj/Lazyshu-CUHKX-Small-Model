#!/usr/bin/env python3
import torch
from torch import nn

from experiments.v97_protected_visual_adapters.model import (
    build_from_source as build_visual_from_source,
)


class TemporalTokenResidualAdapter(nn.Module):
    def __init__(self, source, dimension=192, bottleneck=64):
        super().__init__()
        self.source = source
        self.down = nn.Linear(dimension, bottleneck, bias=False)
        self.normalization = nn.LayerNorm(bottleneck)
        self.context = nn.Conv1d(
            bottleneck,
            bottleneck,
            kernel_size=3,
            padding=1,
            groups=bottleneck,
            bias=False,
        )
        self.context_normalization = nn.LayerNorm(bottleneck)
        self.up = nn.Linear(bottleneck, dimension)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, values):
        tokens = self.source(values)
        residual = torch.nn.functional.gelu(
            self.normalization(self.down(tokens))
        )
        residual = self.context(residual.transpose(1, 2)).transpose(1, 2)
        residual = torch.nn.functional.gelu(
            self.context_normalization(residual)
        )
        return tokens + self.up(residual)

    def adapter_parameters(self):
        for name, parameter in self.named_parameters():
            if not name.startswith("source."):
                yield parameter


def build_from_source(source_path, bottleneck=128, output_adapters=True):
    model, teacher, config = build_visual_from_source(
        source_path,
        bottleneck=bottleneck,
        output_adapters=output_adapters,
    )
    model.base.skeleton = TemporalTokenResidualAdapter(model.base.skeleton)
    model.base.imu = TemporalTokenResidualAdapter(model.base.imu)
    for wrapper in (model.base.skeleton, model.base.imu):
        for parameter in wrapper.adapter_parameters():
            parameter.requires_grad_(True)
    return model, teacher, config

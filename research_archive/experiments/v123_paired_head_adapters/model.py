#!/usr/bin/env python3
import copy

import torch

from experiments.scene_context_v13.adapt_paired_source import (
    checkpoint_state,
    select_trainable,
)
from scripts.evaluate_r2plus1d_checkpoint import build_model


def build_from_source(source_path, bottleneck=128, output_adapters=True):
    del bottleneck, output_adapters
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    state = checkpoint_state(checkpoint)
    model = build_model(config, copy.deepcopy(state))
    teacher = build_model(config, copy.deepcopy(state))
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    select_trainable(model, "heads")
    return model, teacher, config

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


class WideIterativeR2Plus1D(IterativeR2Plus1D):
    def __init__(
        self,
        base,
        iterations=1,
        wide_dimension=768,
        wide_depth=4,
        wide_heads=12,
        feedforward_multiplier=3,
        feedback_temperature=2.0,
        detach_feedback=True,
        residual_scale=1.0,
    ):
        if wide_dimension % wide_heads:
            raise ValueError("wide_dimension must be divisible by wide_heads")
        if wide_depth < 1 or feedforward_multiplier < 1:
            raise ValueError("wide_depth and feedforward_multiplier must be positive")
        super().__init__(
            base,
            iterations=iterations,
            feedback_temperature=feedback_temperature,
            detach_feedback=detach_feedback,
        )
        source_dimension = int(base.normalization.normalized_shape[0])
        classes = int(base.head[-1].out_features)
        self.wide_dimension = wide_dimension
        self.wide_depth = wide_depth
        self.wide_heads = wide_heads
        self.feedforward_multiplier = feedforward_multiplier
        self.residual_scale = residual_scale
        self.stream_projection = nn.Sequential(
            nn.Linear(source_dimension, wide_dimension),
            nn.LayerNorm(wide_dimension),
        )
        self.class_projection = nn.Sequential(
            nn.Linear(source_dimension, wide_dimension),
            nn.LayerNorm(wide_dimension),
        )
        layer = nn.TransformerEncoderLayer(
            wide_dimension,
            nhead=wide_heads,
            dim_feedforward=wide_dimension * feedforward_multiplier,
            dropout=0.15,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.refinement = nn.TransformerEncoder(
            layer,
            num_layers=wide_depth,
            enable_nested_tensor=False,
        )
        self.refinement_normalization = nn.LayerNorm(wide_dimension)
        self.feedback_projection = nn.Linear(classes, wide_dimension)
        self.feedback_normalization = nn.LayerNorm(wide_dimension)
        self.residual_head = nn.Sequential(
            nn.Linear(wide_dimension, wide_dimension),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(wide_dimension, classes),
        )
        nn.init.trunc_normal_(self.feedback_projection.weight, std=0.02)
        nn.init.zeros_(self.feedback_projection.bias)
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def base_prediction(self, encoded, batch_size, device):
        class_token = self.base.class_token.expand(batch_size, -1, -1)
        class_padding = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
        padding_mask = torch.cat([class_padding, encoded.padding_mask], dim=1)
        tokens = torch.cat([class_token, encoded.stream_tokens], dim=1)
        fused = self.base.fusion(tokens, src_key_padding_mask=padding_mask)
        fused_class = self.base.normalization(fused[:, :1])
        logits = self.classify(fused_class[:, 0], encoded)
        return fused_class, logits, padding_mask

    def forward(
        self,
        video,
        skeleton=None,
        imu=None,
        radar=None,
        sensor_mask=None,
        visual_mask=None,
        return_iteration_logits=False,
        return_base_logits=False,
    ):
        encoded = self.encode_once(video, skeleton, imu, radar, sensor_mask, visual_mask)
        base_class, base_logits, padding_mask = self.base_prediction(
            encoded, video.shape[0], video.device
        )
        stream_tokens = self.stream_projection(encoded.stream_tokens)
        class_state = self.class_projection(base_class)
        iteration_logits = []
        previous_probabilities = None
        for step in range(self.iterations):
            if iteration_logits:
                feedback_probabilities = functional.softmax(
                    iteration_logits[-1] / self.feedback_temperature,
                    dim=1,
                )
                if self.detach_feedback:
                    feedback_probabilities = feedback_probabilities.detach()
                feedback = self.feedback_projection(feedback_probabilities).unsqueeze(1)
                class_state = self.feedback_normalization(
                    class_state + self.feedback_gate.sigmoid() * feedback
                )
            tokens = torch.cat([class_state, stream_tokens], dim=1)
            fused = self.refinement(tokens, src_key_padding_mask=padding_mask)
            class_state = self.refinement_normalization(fused[:, :1])
            logits = base_logits + self.residual_scale * self.residual_head(class_state[:, 0])
            iteration_logits.append(logits)
            probabilities = logits.float().softmax(dim=1)
            if self.should_exit(probabilities, previous_probabilities, step):
                break
            previous_probabilities = probabilities
        if return_iteration_logits:
            outputs = (
                iteration_logits[-1],
                encoded.auxiliary,
                torch.stack(iteration_logits, dim=1),
            )
            return outputs + (base_logits,) if return_base_logits else outputs
        return iteration_logits[-1], encoded.auxiliary


def build_from_source(
    source_path,
    iterations,
    wide_dimension=768,
    wide_depth=4,
    wide_heads=12,
    feedforward_multiplier=3,
    feedback_temperature=2.0,
    detach_feedback=True,
    residual_scale=1.0,
):
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if config["mode"] != "shared_multimodal":
        raise ValueError("Wide iterative refinement requires shared_multimodal mode")
    base = build_base_model(config)
    base.load_state_dict(checkpoint_state(checkpoint), strict=True)
    model = WideIterativeR2Plus1D(
        base,
        iterations=iterations,
        wide_dimension=wide_dimension,
        wide_depth=wide_depth,
        wide_heads=wide_heads,
        feedforward_multiplier=feedforward_multiplier,
        feedback_temperature=feedback_temperature,
        detach_feedback=detach_feedback,
        residual_scale=residual_scale,
    )
    return model, config, checkpoint

#!/usr/bin/env python3
from dataclasses import dataclass
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.r2plus1d_models import R2Plus1DClassifier, dequantize_state_dict


@dataclass
class EncodedBatch:
    stream_tokens: torch.Tensor
    padding_mask: torch.Tensor
    video_logits: torch.Tensor
    auxiliary: dict
    expert_logits: torch.Tensor | None
    expert_available: torch.Tensor | None


def checkpoint_state(checkpoint):
    for key in ("model_int4", "model_int7", "model_int6", "model_int8"):
        if key in checkpoint:
            return dequantize_state_dict(checkpoint[key])
    return checkpoint["model"]


def build_base_model(config):
    return R2Plus1DClassifier(
        mode=config["mode"],
        weights=None,
        dropout=float(config.get("dropout", 0.3)),
        input_channels=int(config.get("input_channels", 7)),
        mask_joint_visual=bool(config.get("mask_joint_visual", False)),
        use_radar=bool(config.get("use_radar", False)),
        skeleton_representation=config.get("skeleton_representation", "coordinates"),
        mixstyle_probability=float(config.get("mixstyle_probability", 0.0)),
        mixstyle_alpha=float(config.get("mixstyle_alpha", 0.1)),
        fusion_depth=int(config.get("fusion_depth", 2)),
        use_temporal_embedding=bool(config.get("use_temporal_embedding", False)),
        use_phase_temporal_embedding=bool(config.get("use_phase_temporal_embedding", False)),
        fusion_output=config.get("fusion_output", "residual"),
        visual_motion_channels=bool(config.get("visual_motion_channels", False)),
        visual_strategy=config.get("visual_strategy", "sequential"),
        per_visual_auxiliary=bool(config.get("per_visual_auxiliary", False)),
        environment_moe=float(config.get("environment_auxiliary_weight", 0.0)) > 0,
        visual_multiscale_tokens=bool(config.get("visual_multiscale_tokens", False)),
        normalize_modality_tokens=bool(config.get("normalize_modality_tokens", False)),
    )


def build_from_source(source_path, iterations, feedback_temperature=2.0, detach_feedback=True):
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if config["mode"] != "shared_multimodal":
        raise ValueError("V24 iterative refinement requires shared_multimodal mode")
    base = build_base_model(config)
    state = checkpoint_state(checkpoint)
    if any(name.startswith("base.") for name in state):
        model = IterativeR2Plus1D(
            base,
            iterations=iterations,
            feedback_temperature=feedback_temperature,
            detach_feedback=detach_feedback,
        )
        model.load_state_dict(state, strict=True)
    else:
        base.load_state_dict(state, strict=True)
        model = IterativeR2Plus1D(
            base,
            iterations=iterations,
            feedback_temperature=feedback_temperature,
            detach_feedback=detach_feedback,
        )
    return model, config, checkpoint


class IterativeR2Plus1D(nn.Module):
    def __init__(
        self,
        base,
        iterations=1,
        feedback_temperature=2.0,
        detach_feedback=True,
        early_exit_confidence=None,
        early_exit_delta=None,
        early_exit_min_iterations=2,
    ):
        super().__init__()
        if base.mode != "shared_multimodal":
            raise ValueError("Only shared_multimodal models are supported")
        if not 1 <= iterations <= 3:
            raise ValueError("iterations must be in [1, 3]")
        if feedback_temperature <= 0:
            raise ValueError("feedback_temperature must be positive")
        self.base = base
        self.iterations = iterations
        self.feedback_temperature = feedback_temperature
        self.detach_feedback = detach_feedback
        self.early_exit_confidence = early_exit_confidence
        self.early_exit_delta = early_exit_delta
        self.early_exit_min_iterations = early_exit_min_iterations
        dimension = int(base.normalization.normalized_shape[0])
        classes = int(base.head[-1].out_features)
        self.feedback_projection = nn.Linear(classes, dimension)
        self.feedback_normalization = nn.LayerNorm(dimension)
        self.feedback_gate = nn.Parameter(torch.tensor(-2.0))
        nn.init.trunc_normal_(self.feedback_projection.weight, std=0.02)
        nn.init.zeros_(self.feedback_projection.bias)

    @property
    def video(self):
        return self.base.video

    @property
    def video_head(self):
        return self.base.video_head

    @property
    def normalization(self):
        return self.base.normalization

    def extra_parameter_count(self):
        return sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if not name.startswith("base.")
        )

    def encode_once(self, video, skeleton, imu, radar, sensor_mask, visual_mask):
        base = self.base
        if video.shape[1] != 7:
            raise ValueError("Shared visual mode requires seven Depth/Thermal/IR channels")
        if skeleton is None or imu is None or sensor_mask is None:
            raise ValueError("Skeleton, IMU, and sensor masks are required")

        visual_inputs = [video[:, :3], video[:, 3:6], video[:, 6:7].expand(-1, 3, -1, -1, -1)]
        visual_inputs = [base.visual_representation(values) for values in visual_inputs]
        if base.visual_strategy in ("joint", "separate_stem_joint"):
            encoded_visual = base.video.forward_joint(
                visual_inputs, return_multiscale=base.visual_multiscale_tokens
            )
        else:
            encoded_visual = [
                base.video(
                    values,
                    modality=index,
                    return_multiscale=base.visual_multiscale_tokens,
                )
                for index, values in enumerate(visual_inputs)
            ]
        if base.visual_multiscale_tokens:
            video_intermediate_maps = [values[0] for values in encoded_visual]
            video_maps = [values[1] for values in encoded_visual]
        else:
            video_intermediate_maps = None
            video_maps = encoded_visual

        video_features = torch.stack(
            [values.mean(dim=(2, 3, 4)) for values in video_maps], dim=1
        )
        stream_logits = base.video_head(video_features)
        if visual_mask is None:
            visual_mask = torch.ones(video.shape[0], 3, device=video.device)
        visual_weights = visual_mask / visual_mask.sum(dim=1, keepdim=True).clamp_min(1)
        video_logits = (stream_logits * visual_weights.unsqueeze(-1)).sum(dim=1)

        skeleton_tokens = base.skeleton(skeleton)
        imu_tokens = base.imu(imu)
        radar_tokens = base.radar(radar) if base.use_radar else None
        video_streams = []
        for index, values in enumerate(video_maps):
            final_tokens = base.video_projection(values.mean(dim=(3, 4)).transpose(1, 2))
            if base.visual_multiscale_tokens:
                intermediate_tokens = base.video_intermediate_projection(
                    video_intermediate_maps[index].mean(dim=(3, 4)).transpose(1, 2)
                )
                final_tokens = torch.cat([intermediate_tokens, final_tokens], dim=1)
            video_streams.append(final_tokens)
        streams = video_streams + [skeleton_tokens, imu_tokens]
        if radar_tokens is not None:
            streams.append(radar_tokens)
        for index, tokens in enumerate(streams):
            if base.normalize_modality_tokens:
                tokens = base.modality_normalizations[index](tokens)
            streams[index] = tokens + base.modality_embedding[:, index]
            if base.use_phase_temporal_embedding:
                phase_embedding = functional.interpolate(
                    base.temporal_embedding.transpose(1, 2),
                    size=tokens.shape[1],
                    mode="linear",
                    align_corners=True,
                ).transpose(1, 2)
                streams[index] = streams[index] + phase_embedding
            elif base.use_temporal_embedding:
                streams[index] = streams[index] + base.temporal_embedding[:, :tokens.shape[1]]

        video_masks = [
            (~visual_mask[:, index].bool()).unsqueeze(1).expand(-1, tokens.shape[1])
            for index, tokens in enumerate(video_streams)
        ]
        skeleton_mask = (~sensor_mask[:, 0].bool()).unsqueeze(1).expand(
            -1, skeleton_tokens.shape[1]
        )
        imu_mask = (~sensor_mask[:, 1].bool()).unsqueeze(1).expand(-1, imu_tokens.shape[1])
        radar_masks = (
            [(~sensor_mask[:, 2].bool()).unsqueeze(1).expand(-1, radar_tokens.shape[1])]
            if radar_tokens is not None
            else []
        )
        stream_tokens = torch.cat(streams, dim=1)
        padding_mask = torch.cat(video_masks + [skeleton_mask, imu_mask] + radar_masks, dim=1)

        auxiliary = {
            ("video_streams" if base.per_visual_auxiliary else "video"):
                stream_logits if base.per_visual_auxiliary else video_logits,
            "skeleton": base.skeleton_head(skeleton_tokens.mean(dim=1)),
            "imu": base.imu_head(imu_tokens.mean(dim=1)),
        }
        if radar_tokens is not None:
            auxiliary["radar"] = base.radar_head(radar_tokens.mean(dim=1))

        expert_logits = None
        expert_available = None
        if base.fusion_output in ("gated", "stream_gated"):
            if base.fusion_output == "stream_gated":
                expert_logits = torch.cat(
                    [stream_logits]
                    + [auxiliary[name].unsqueeze(1) for name in ("skeleton", "imu")]
                    + ([auxiliary["radar"].unsqueeze(1)] if radar_tokens is not None else []),
                    dim=1,
                )
                expert_available = torch.cat(
                    [visual_mask.bool(), sensor_mask[:, :2 + int(radar_tokens is not None)].bool()],
                    dim=1,
                )
            else:
                expert_logits = torch.stack(list(auxiliary.values()), dim=1)
                video_available = visual_mask.bool().any(dim=1)
                expert_available = torch.cat(
                    [
                        video_available.unsqueeze(1),
                        sensor_mask[:, :2 + int(radar_tokens is not None)].bool(),
                    ],
                    dim=1,
                )
        return EncodedBatch(
            stream_tokens=stream_tokens,
            padding_mask=padding_mask,
            video_logits=video_logits,
            auxiliary=auxiliary,
            expert_logits=expert_logits,
            expert_available=expert_available,
        )

    def classify(self, fused_class, encoded):
        base = self.base
        if base.fusion_output in ("gated", "stream_gated"):
            gate_logits = base.expert_gate(fused_class).reshape(
                fused_class.shape[0], encoded.expert_logits.shape[1], 40
            )
            gate_logits = gate_logits.masked_fill(~encoded.expert_available.unsqueeze(2), -1e4)
            output = (
                gate_logits.softmax(dim=1) * encoded.expert_logits
            ).sum(dim=1) + base.head(fused_class)
        else:
            output = encoded.video_logits + base.head(fused_class)
        if base.environment_moe:
            environment_logits = base.environment_gate(fused_class)
            environment_weights = environment_logits.softmax(dim=1)
            environment_residuals = base.environment_residual(fused_class).reshape(-1, 2, 40)
            output = output + (
                environment_weights.unsqueeze(2) * environment_residuals
            ).sum(dim=1)
            encoded.auxiliary["environment"] = environment_logits
        return output

    def should_exit(self, probabilities, previous_probabilities, step):
        if self.training or step + 1 < self.early_exit_min_iterations:
            return False
        confidence_exit = (
            self.early_exit_confidence is not None
            and bool((probabilities.amax(dim=1) >= self.early_exit_confidence).all())
        )
        delta_exit = (
            self.early_exit_delta is not None
            and previous_probabilities is not None
            and bool(
                ((probabilities - previous_probabilities).abs().amax(dim=1) <= self.early_exit_delta).all()
            )
        )
        return confidence_exit or delta_exit

    def forward(
        self,
        video,
        skeleton=None,
        imu=None,
        radar=None,
        sensor_mask=None,
        visual_mask=None,
        return_iteration_logits=False,
    ):
        encoded = self.encode_once(video, skeleton, imu, radar, sensor_mask, visual_mask)
        batch_size = video.shape[0]
        class_state = self.base.class_token.expand(batch_size, -1, -1)
        class_padding = torch.zeros(batch_size, 1, dtype=torch.bool, device=video.device)
        padding_mask = torch.cat([class_padding, encoded.padding_mask], dim=1)
        iteration_logits = []
        previous_probabilities = None
        for step in range(self.iterations):
            if iteration_logits:
                feedback_probabilities = functional.softmax(
                    iteration_logits[-1] / self.feedback_temperature, dim=1
                )
                if self.detach_feedback:
                    feedback_probabilities = feedback_probabilities.detach()
                feedback = self.feedback_projection(feedback_probabilities).unsqueeze(1)
                class_state = self.feedback_normalization(
                    class_state + self.feedback_gate.sigmoid() * feedback
                )
            tokens = torch.cat([class_state, encoded.stream_tokens], dim=1)
            fused = self.base.fusion(tokens, src_key_padding_mask=padding_mask)
            class_state = self.base.normalization(fused[:, :1])
            logits = self.classify(class_state[:, 0], encoded)
            iteration_logits.append(logits)
            probabilities = logits.float().softmax(dim=1)
            if self.should_exit(probabilities, previous_probabilities, step):
                break
            previous_probabilities = probabilities
        if return_iteration_logits:
            return iteration_logits[-1], encoded.auxiliary, torch.stack(iteration_logits, dim=1)
        return iteration_logits[-1], encoded.auxiliary

#!/usr/bin/env python3
import copy
from pathlib import Path
import sys
import unittest

import torch
from torch import nn


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from iterative_model import IterativeR2Plus1D


class SpyVideo(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward_joint(self, videos, return_multiscale=False):
        self.calls += 1
        outputs = []
        channel = torch.linspace(-0.2, 0.2, 512).reshape(1, 512, 1, 1, 1)
        for values in videos:
            pooled = values.mean(dim=(1, 2, 3, 4), keepdim=True)
            outputs.append((pooled + channel).expand(-1, -1, 4, 2, 2))
        return outputs


class SpyTokenEncoder(nn.Module):
    def __init__(self, input_dimension, output_dimension, steps=16):
        super().__init__()
        self.calls = 0
        self.steps = steps
        self.projection = nn.Linear(input_dimension, output_dimension)

    def forward(self, values):
        self.calls += 1
        values = values[:, :self.steps]
        return self.projection(values)


class FakeBase(nn.Module):
    def __init__(self, dimension=24):
        super().__init__()
        self.mode = "shared_multimodal"
        self.visual_strategy = "joint"
        self.visual_multiscale_tokens = False
        self.normalize_modality_tokens = False
        self.use_phase_temporal_embedding = False
        self.use_temporal_embedding = False
        self.use_radar = False
        self.per_visual_auxiliary = False
        self.fusion_output = "residual"
        self.environment_moe = False
        self.video = SpyVideo()
        self.video_head = nn.Linear(512, 40)
        self.video_projection = nn.Linear(512, dimension)
        self.skeleton = SpyTokenEncoder(6, dimension)
        self.imu = SpyTokenEncoder(6, dimension)
        self.class_token = nn.Parameter(torch.randn(1, 1, dimension) * 0.02)
        self.modality_embedding = nn.Parameter(torch.randn(1, 5, 1, dimension) * 0.02)
        layer = nn.TransformerEncoderLayer(
            dimension,
            nhead=6,
            dim_feedforward=dimension * 3,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.normalization = nn.LayerNorm(dimension)
        self.head = nn.Sequential(nn.Linear(dimension, dimension), nn.GELU(), nn.Linear(dimension, 40))
        self.skeleton_head = nn.Linear(dimension, 40)
        self.imu_head = nn.Linear(dimension, 40)

    def visual_representation(self, values):
        return values

    def forward(self, video, skeleton, imu, sensor_mask, visual_mask, radar=None):
        visual_inputs = [video[:, :3], video[:, 3:6], video[:, 6:7].expand(-1, 3, -1, -1, -1)]
        video_maps = self.video.forward_joint(visual_inputs)
        video_features = torch.stack([values.mean(dim=(2, 3, 4)) for values in video_maps], dim=1)
        stream_logits = self.video_head(video_features)
        visual_weights = visual_mask / visual_mask.sum(dim=1, keepdim=True).clamp_min(1)
        video_logits = (stream_logits * visual_weights.unsqueeze(-1)).sum(dim=1)
        skeleton_tokens = self.skeleton(skeleton)
        imu_tokens = self.imu(imu)
        video_streams = [
            self.video_projection(values.mean(dim=(3, 4)).transpose(1, 2)) for values in video_maps
        ]
        streams = video_streams + [skeleton_tokens, imu_tokens]
        streams = [
            values + self.modality_embedding[:, index] for index, values in enumerate(streams)
        ]
        tokens = torch.cat([self.class_token.expand(video.shape[0], -1, -1)] + streams, dim=1)
        visual_masks = [
            (~visual_mask[:, index].bool()).unsqueeze(1).expand(-1, values.shape[1])
            for index, values in enumerate(video_streams)
        ]
        skeleton_mask = (~sensor_mask[:, 0].bool()).unsqueeze(1).expand(-1, skeleton_tokens.shape[1])
        imu_mask = (~sensor_mask[:, 1].bool()).unsqueeze(1).expand(-1, imu_tokens.shape[1])
        padding = torch.cat(
            [torch.zeros(video.shape[0], 1, dtype=torch.bool)]
            + visual_masks
            + [skeleton_mask, imu_mask],
            dim=1,
        )
        fused = self.fusion(tokens, src_key_padding_mask=padding)
        logits = video_logits + self.head(self.normalization(fused[:, 0]))
        return logits, {
            "video": video_logits,
            "skeleton": self.skeleton_head(skeleton_tokens.mean(dim=1)),
            "imu": self.imu_head(imu_tokens.mean(dim=1)),
        }


def inputs():
    torch.manual_seed(7)
    return {
        "video": torch.randn(2, 7, 4, 8, 8),
        "skeleton": torch.randn(2, 16, 6),
        "imu": torch.randn(2, 16, 6),
        "sensor_mask": torch.tensor([[1, 1], [1, 0]], dtype=torch.float32),
        "visual_mask": torch.tensor([[1, 1, 1], [1, 0, 1]], dtype=torch.float32),
    }


class IterativeModelTest(unittest.TestCase):
    def test_k1_exactly_matches_non_iterative_reference(self):
        base = FakeBase().eval()
        iterative = IterativeR2Plus1D(base, iterations=1).eval()
        batch = inputs()
        with torch.no_grad():
            reference, reference_auxiliary = base(**batch)
            actual, actual_auxiliary, steps = iterative(**batch, return_iteration_logits=True)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        for name in reference_auxiliary:
            torch.testing.assert_close(actual_auxiliary[name], reference_auxiliary[name], rtol=0, atol=0)
        self.assertEqual(tuple(steps.shape), (2, 1, 40))

    def test_k3_encoders_run_once_and_fusion_is_reused(self):
        base = FakeBase().eval()
        iterative = IterativeR2Plus1D(base, iterations=3).eval()
        fusion_module_ids = []
        handle = base.fusion.register_forward_hook(
            lambda module, inputs, output: fusion_module_ids.append(id(module))
        )
        with torch.no_grad():
            logits, _, steps = iterative(**inputs(), return_iteration_logits=True)
        handle.remove()
        self.assertEqual(base.video.calls, 1)
        self.assertEqual(base.skeleton.calls, 1)
        self.assertEqual(base.imu.calls, 1)
        self.assertEqual(fusion_module_ids, [id(base.fusion)] * 3)
        self.assertEqual(tuple(logits.shape), (2, 40))
        self.assertEqual(tuple(steps.shape), (2, 3, 40))

    def test_k_changes_no_parameter_count(self):
        state = FakeBase().state_dict()
        models = []
        for iterations in (1, 2, 3):
            base = FakeBase()
            base.load_state_dict(copy.deepcopy(state))
            models.append(IterativeR2Plus1D(base, iterations=iterations))
        counts = [sum(parameter.numel() for parameter in model.parameters()) for model in models]
        self.assertEqual(counts, [counts[0]] * 3)
        dimension = 24
        expected = 40 * dimension + dimension + 2 * dimension + 1
        self.assertEqual(models[0].extra_parameter_count(), expected)

    def test_feedback_receives_gradient_and_can_early_exit(self):
        model = IterativeR2Plus1D(FakeBase(), iterations=3)
        _, _, steps = model(**inputs(), return_iteration_logits=True)
        steps.mean().backward()
        self.assertIsNotNone(model.feedback_projection.weight.grad)
        model.eval()
        model.early_exit_confidence = 0.0
        with torch.no_grad():
            _, _, exited_steps = model(**inputs(), return_iteration_logits=True)
        self.assertEqual(exited_steps.shape[1], 2)


if __name__ == "__main__":
    unittest.main()

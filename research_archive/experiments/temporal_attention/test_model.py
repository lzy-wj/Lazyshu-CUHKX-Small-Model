#!/usr/bin/env python3
import sys
from pathlib import Path

import torch
from torch import nn


EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EXPERIMENT_ROOT))

from model import RelativeTimeFusion, TemporalAttentionR2Plus1DClassifier
from src.r2plus1d_models import R2Plus1DClassifier


class FakeVideo(nn.Module):
    def forward(self, video):
        pooled = video.mean(dim=(1, 2, 3, 4), keepdim=True)
        return pooled.expand(-1, 512, 4, 2, 2)


class FakeTokens(nn.Module):
    def forward(self, values):
        pooled = values.mean(dim=(1, 2), keepdim=True)
        return pooled.expand(-1, 16, 192)


def main():
    torch.manual_seed(7)
    fusion = RelativeTimeFusion().eval()
    class_token = torch.randn(2, 1, 192)
    streams = [torch.randn(2, 4, 192), torch.randn(2, 16, 192), torch.randn(2, 16, 192)]
    padding = torch.zeros(2, 37, dtype=torch.bool)
    padding[1, 5:21] = True
    output = fusion(class_token, streams, padding)
    assert output.shape == (2, 37, 192)
    assert torch.isfinite(output).all()

    candidate = TemporalAttentionR2Plus1DClassifier(weights=None).eval()
    baseline = R2Plus1DClassifier(mode="multimodal", weights=None).eval()
    candidate_parameters = sum(parameter.numel() for parameter in candidate.parameters())
    baseline_parameters = sum(parameter.numel() for parameter in baseline.parameters())
    assert candidate_parameters - baseline_parameters == 325
    candidate.video = FakeVideo()
    candidate.skeleton = FakeTokens()
    candidate.imu = FakeTokens()
    video = torch.randn(2, 4, 32, 8, 8)
    skeleton = torch.randn(2, 64, 153)
    imu = torch.randn(2, 64, 85)
    sensor_mask = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    visual_mask = torch.ones(2, 2)
    logits, auxiliary = candidate(video, skeleton, imu, sensor_mask, visual_mask)
    assert logits.shape == (2, 40)
    assert set(auxiliary) == {"video", "skeleton", "imu"}
    changed = skeleton.clone()
    changed[1].add_(1000)
    changed_logits, _ = candidate(video, changed, imu, sensor_mask, visual_mask)
    assert torch.allclose(logits[1], changed_logits[1], atol=1e-5, rtol=1e-5)
    print({
        "shape": list(logits.shape),
        "mask_invariance": True,
        "baseline_parameters": baseline_parameters,
        "candidate_parameters": candidate_parameters,
        "parameter_delta": candidate_parameters - baseline_parameters,
    })


if __name__ == "__main__":
    main()

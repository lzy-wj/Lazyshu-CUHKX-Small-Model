#!/usr/bin/env python3
import sys
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.train_r2plus1d import (
    CrossUserContrastiveQueue,
    GradientReversal,
    prepare_batch,
    temporal_resample,
)
from src.r2plus1d_models import R2Plus1DClassifier


def audit_gradient_reversal():
    values = torch.tensor([[1.0, -2.0]], requires_grad=True)
    GradientReversal.apply(values, 0.25).square().sum().backward()
    torch.testing.assert_close(values.grad, -0.5 * values.detach())


def audit_cross_user_contrastive():
    features = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], requires_grad=True
    )
    labels = torch.tensor([0, 0, 1, 1])
    users = torch.tensor([1, 2, 1, 2])
    queue = CrossUserContrastiveQueue(2, 8, 0.1, torch.device("cpu"))
    loss, active = queue.loss(features, labels, users)
    assert active == 4 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(features.grad).all()
    queue.update(features, labels, users)
    assert len(queue.labels) == 4
    empty_queue = CrossUserContrastiveQueue(2, 8, 0.1, torch.device("cpu"))
    no_positive, active = empty_queue.loss(features.detach(), labels, torch.ones_like(users))
    assert active == 0 and float(no_positive) == 0
    balanced = CrossUserContrastiveQueue(
        2, 16, 0.1, torch.device("cpu"), class_cap=4, class_user_cap=2
    )
    for _ in range(4):
        balanced.update(features.detach(), labels, users)
    for label in labels.unique():
        assert int((balanced.labels == label).sum()) <= 4
        for user in users.unique():
            assert int(((balanced.labels == label) & (balanced.users == user)).sum()) <= 2


def audit_visual_channel_dropout():
    batch = {
        "video": torch.ones(4, 2, 4, 3, 3),
        "label": torch.arange(4),
        "index": torch.arange(4),
        "visual_mask": torch.ones(4, 2),
        "skeleton": torch.ones(4, 2, 3),
        "imu": torch.ones(4, 2, 3),
        "sensor_mask": torch.ones(4, 2),
    }
    torch.manual_seed(7)
    output = prepare_batch(
        batch, torch.device("cpu"), multimodal=True, training=True, visual_channel_dropout=1.0
    )
    assert torch.equal(output["visual_channel_drop"].sum(dim=1), torch.ones(4, dtype=torch.long))
    rgb_zero = (output["video"][:, :3] == 0).flatten(1).all(1)
    ir_zero = (output["video"][:, 3:] == 0).flatten(1).all(1)
    assert torch.equal(rgb_zero ^ ir_zero, torch.ones(4, dtype=torch.bool))
    batch["video"].fill_(1)
    clean = prepare_batch(
        batch, torch.device("cpu"), multimodal=True, training=False, visual_channel_dropout=1.0
    )
    assert "visual_channel_drop" not in clean
    assert (clean["video"] == 1).all()


def audit_structured_modality_dropout():
    batch = {
        "video": torch.ones(6, 2, 4, 3, 3),
        "label": torch.arange(6),
        "index": torch.arange(6),
        "visual_mask": torch.ones(6, 2),
        "skeleton": torch.ones(6, 2, 3),
        "imu": torch.ones(6, 2, 3),
        "sensor_mask": torch.ones(6, 2),
    }
    torch.manual_seed(11)
    output = prepare_batch(
        batch,
        torch.device("cpu"),
        multimodal=True,
        training=True,
        structured_modality_dropout=1.0,
    )
    assert torch.equal(output["branch_drop"].sum(dim=1), torch.full((6,), 2))
    retained = torch.cat(
        [output["visual_mask"].bool().any(dim=1, keepdim=True), output["sensor_mask"].bool()], dim=1
    )
    assert torch.equal(retained.sum(dim=1), torch.ones(6, dtype=torch.long))
    for position, drops in enumerate(output["branch_drop"]):
        if drops[0]:
            assert not output["video"][position].any()
        if drops[1]:
            assert not output["skeleton"][position].any()
        if drops[2]:
            assert not output["imu"][position].any()


def audit_joint_visual_mask():
    class FakeVideo(nn.Module):
        def forward(self, video):
            return torch.ones(video.shape[0], 512, 2, 1, 1)

    class FakeTokens(nn.Module):
        def forward(self, values):
            return torch.zeros(values.shape[0], 2, 192)

    class FakeFusion(nn.Module):
        def forward(self, values, src_key_padding_mask=None):
            return values

    model = R2Plus1DClassifier(mode="multimodal", weights=None, mask_joint_visual=True).eval()
    model.video = FakeVideo()
    model.skeleton = FakeTokens()
    model.imu = FakeTokens()
    model.fusion = FakeFusion()
    with torch.no_grad():
        model.video_head[-1].weight.fill_(0.01)
        model.video_head[-1].bias.zero_()
    inputs = {
        "video": torch.ones(2, 4, 2, 3, 3),
        "skeleton": torch.ones(2, 2, 3),
        "imu": torch.ones(2, 2, 3),
        "sensor_mask": torch.ones(2, 2),
    }
    _, present = model(**inputs, visual_mask=torch.ones(2, 2))
    _, missing = model(**inputs, visual_mask=torch.zeros(2, 2))
    assert present["video"].abs().sum() > 0
    assert not missing["video"].any()


def audit_temporal_resampling():
    values = torch.arange(12).reshape(2, 6, 1)
    identity = temporal_resample(values, torch.ones(2), 1)
    torch.testing.assert_close(identity, values)
    video = values.transpose(1, 2).unsqueeze(-1).unsqueeze(-1)
    warped_video = temporal_resample(video, torch.tensor([0.8, 1.2]), 2)
    warped_sensor = temporal_resample(values, torch.tensor([0.8, 1.2]), 1)
    torch.testing.assert_close(warped_video.squeeze(-1).squeeze(-1).transpose(1, 2), warped_sensor)


if __name__ == "__main__":
    audit_gradient_reversal()
    audit_cross_user_contrastive()
    audit_visual_channel_dropout()
    audit_structured_modality_dropout()
    audit_joint_visual_mask()
    audit_temporal_resampling()
    print("V11 method audit passed")

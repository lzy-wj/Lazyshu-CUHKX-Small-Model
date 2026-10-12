#!/usr/bin/env python3
import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import FOLDS, ROOT, get_indices, get_records
from src.r2plus1d_data import R2Plus1DDataset
from src.r2plus1d_models import R2Plus1DClassifier, quantize_state_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--fold", type=int, choices=(-1, 0, 1), required=True)
    parser.add_argument("--mode", choices=("visual", "multimodal", "shared_multimodal"), default="visual")
    parser.add_argument("--initialization", choices=("scratch", "kinetics"), default="kinetics")
    parser.add_argument("--weights", type=Path, default=ROOT / ".tools/checkpoints/r2plus1d_34_clip32_ft_kinetics_from_ig65m-ade133f1.pth")
    parser.add_argument("--init-visual", type=Path)
    parser.add_argument("--init-model", type=Path)
    parser.add_argument("--init-skeleton", type=Path)
    parser.add_argument("--freeze-visual", action="store_true")
    parser.add_argument("--freeze-inherited", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/processed/r2plus1d_yolo_16_128")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--backbone-lr-factor", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=float, default=2)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--auxiliary-weight", type=float, default=0.15)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--class-balance-power", type=float, default=0.0)
    parser.add_argument("--class-balanced-sampling", action="store_true")
    parser.add_argument("--class-user-balanced-sampling", action="store_true")
    parser.add_argument("--group-dro-eta", type=float, default=0.0)
    parser.add_argument("--hierarchical-loss-weight", type=float, default=0.0)
    parser.add_argument("--balanced-softmax-temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--limit-train-batches", type=int, default=0)
    parser.add_argument("--limit-valid-batches", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--visual-channel-dropout", type=float, default=0.0)
    parser.add_argument("--structured-modality-dropout", type=float, default=0.0)
    parser.add_argument("--masked-modality-distillation-weight", type=float, default=0.0)
    parser.add_argument("--masked-modality-distillation-temperature", type=float, default=2.0)
    parser.add_argument("--cross-user-contrastive-weight", type=float, default=0.0)
    parser.add_argument("--cross-environment-contrastive-weight", type=float, default=0.0)
    parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--contrastive-queue-size", type=int, default=2048)
    parser.add_argument("--contrastive-class-cap", type=int, default=0)
    parser.add_argument("--contrastive-class-user-cap", type=int, default=0)
    parser.add_argument("--contrastive-warmup-items", type=int, default=0)
    parser.add_argument("--subject-adversarial-weight", type=float, default=0.0)
    parser.add_argument("--freeze-backbone-bn", action="store_true")
    parser.add_argument("--backbone-bn-momentum", type=float, default=-1.0)
    parser.add_argument("--temporal-scale-jitter", type=float, default=0.0)
    parser.add_argument("--blend-teacher", type=Path)
    parser.add_argument("--blend-loss-weight", type=float, default=0.0)
    parser.add_argument("--blend-base-weight", type=float, default=0.4)
    parser.add_argument("--blend-r2-temperature", type=float, default=2.0)
    parser.add_argument("--save-epoch-checkpoints", action="store_true")
    parser.add_argument("--use-radar", action="store_true")
    parser.add_argument(
        "--radar-representation",
        choices=("statistics", "points"),
        default="statistics",
    )
    parser.add_argument(
        "--imu-representation",
        choices=("world", "relative", "combined", "world_magnetic", "world_official"),
        default="world",
    )
    parser.add_argument("--imu-statistics-residual", action="store_true")
    parser.add_argument("--imu-statistics-logit-residual", action="store_true")
    parser.add_argument("--imu-device-attention-residual", action="store_true")
    parser.add_argument("--photometric-jitter", type=float, default=0.0)
    parser.add_argument("--photometric-gamma", type=float, default=0.0)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument(
        "--skeleton-representation", choices=("coordinates", "centered", "invariant"),
        default="coordinates",
    )
    parser.add_argument(
        "--skeleton-encoder", choices=("tcn", "residual_invariant"), default="tcn"
    )
    parser.add_argument("--canonical-pose", action="store_true")
    parser.add_argument("--mixstyle-probability", type=float, default=0.0)
    parser.add_argument("--mixstyle-alpha", type=float, default=0.1)
    parser.add_argument("--fusion-depth", type=int, default=2)
    parser.add_argument(
        "--fusion-output", choices=("residual", "gated", "stream_gated"), default="residual"
    )
    parser.add_argument("--horizontal-flip-probability", type=float, default=0.5)
    parser.add_argument("--synchronized-horizontal-flip", action="store_true")
    parser.add_argument("--skeleton-scale-randomization", action="store_true")
    parser.add_argument("--use-temporal-embedding", action="store_true")
    parser.add_argument("--use-phase-temporal-embedding", action="store_true")
    parser.add_argument("--per-visual-auxiliary", action="store_true")
    parser.add_argument("--environment-auxiliary-weight", type=float, default=0.0)
    parser.add_argument("--visual-multiscale-tokens", action="store_true")
    parser.add_argument("--normalize-modality-tokens", action="store_true")
    parser.add_argument("--visual-motion-channels", action="store_true")
    parser.add_argument(
        "--visual-strategy",
        choices=(
            "sequential", "joint", "modality_bn", "separate_stem_joint", "joint_film",
            "modality_bn_film",
        ),
        default="sequential",
    )
    parser.add_argument("--allow-method-combination", action="store_true")
    parser.add_argument("--training-labels", default="")
    parser.add_argument("--selection-labels", default="")
    parser.add_argument("--training-environment", type=int, choices=(-1, 0, 1), default=-1)
    parser.add_argument("--selection-environment", type=int, choices=(-1, 0, 1), default=-1)
    parser.add_argument("--validation-users", default="")
    parser.add_argument("--training-minimum-modalities", type=int, default=0)
    parser.add_argument("--selection-minimum-modalities", type=int, default=0)
    parser.add_argument("--focus-labels", default="")
    parser.add_argument("--focus-weight", type=float, default=1.0)
    parser.add_argument("--swanlab-mode", choices=("cloud", "local", "disabled"), default="cloud")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def learning_rate(step, total_steps, warmup_steps, maximum):
    if step < warmup_steps:
        return maximum * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return maximum * 0.5 * (1 + math.cos(math.pi * min(progress, 1)))


def temporal_resample(values, scales, temporal_dimension):
    if temporal_dimension != 1:
        values = values.transpose(1, temporal_dimension)
    steps = values.shape[1]
    normalized = torch.linspace(-0.5, 0.5, steps, device=values.device)
    source = (normalized.unsqueeze(0) * scales.unsqueeze(1) + 0.5).clamp(0, 1)
    indices = source.mul(steps - 1).round().long()
    gather = indices.reshape(len(values), steps, *([1] * (values.ndim - 2))).expand_as(values)
    output = torch.gather(values, 1, gather)
    return output if temporal_dimension == 1 else output.transpose(1, temporal_dimension)


def prepare_batch(
    batch,
    device,
    multimodal,
    training=False,
    visual_channel_dropout=0.0,
    structured_modality_dropout=0.0,
    temporal_scale_jitter=0.0,
    default_modality_dropout=True,
):
    output = {
        "video": batch["video"].to(device, non_blocking=True).permute(0, 2, 1, 3, 4),
        "label": batch["label"].to(device, non_blocking=True),
        "index": batch["index"],
        "visual_mask": batch["visual_mask"].to(device, non_blocking=True),
    }
    if multimodal:
        output.update(
            {
                "skeleton": batch["skeleton"].to(device, non_blocking=True),
                "imu": batch["imu"].to(device, non_blocking=True),
                "sensor_mask": batch["sensor_mask"].to(device, non_blocking=True),
            }
        )
        if "radar" in batch:
            output["radar"] = batch["radar"].to(device, non_blocking=True)
        if training and structured_modality_dropout > 0:
            natural_visual = batch["visual_mask"].to(device, non_blocking=True).bool().any(dim=1)
            natural_branches = torch.cat(
                [natural_visual.unsqueeze(1), output["sensor_mask"].bool()], dim=1
            )
            branch_drop = (
                torch.rand(natural_branches.shape, device=device) < structured_modality_dropout
            ) & natural_branches
            retained = natural_branches & ~branch_drop
            restore = retained.sum(dim=1) == 0
            restore_scores = torch.rand(natural_branches.shape, device=device).masked_fill(
                ~natural_branches, 2
            )
            restored_branch = restore_scores.argmin(dim=1)
            branch_drop[torch.arange(len(branch_drop), device=device), restored_branch] &= ~restore
            output["video"].masked_fill_(branch_drop[:, 0, None, None, None, None], 0)
            output["skeleton"].masked_fill_(
                branch_drop[:, 1].reshape(len(branch_drop), *([1] * (output["skeleton"].ndim - 1))), 0
            )
            output["imu"].masked_fill_(
                branch_drop[:, 2].reshape(len(branch_drop), *([1] * (output["imu"].ndim - 1))), 0
            )
            output["visual_mask"] = output["visual_mask"] * (~branch_drop[:, :1])
            output["sensor_mask"] = output["sensor_mask"] * (~branch_drop[:, 1:])
            output["branch_drop"] = branch_drop
        elif training and default_modality_dropout:
            drop = torch.rand(output["sensor_mask"].shape, device=device) < 0.1
            output["sensor_mask"] = output["sensor_mask"] * (~drop)
            visual_drop = torch.rand(output["visual_mask"].shape, device=device) < 0.1
            dropped_visual = output["visual_mask"] * (~visual_drop)
            empty = dropped_visual.sum(dim=1) == 0
            dropped_visual[empty] = output["visual_mask"][empty]
            output["visual_mask"] = dropped_visual
        if training and visual_channel_dropout > 0:
            available_visual = batch["visual_mask"].to(device, non_blocking=True).bool()
            eligible = available_visual.sum(dim=1) >= 2
            apply_channel_dropout = (
                (torch.rand(len(available_visual), device=device) < visual_channel_dropout) & eligible
            )
            selected = torch.rand(available_visual.shape, device=device).masked_fill(~available_visual, 2).argmin(1)
            channel_drop = torch.zeros_like(available_visual)
            channel_drop[torch.arange(len(channel_drop), device=device), selected] = apply_channel_dropout
            output["visual_mask"] = output["visual_mask"] * (~channel_drop)
            visual_streams = available_visual.shape[1]
            for stream in range(visual_streams):
                start = stream * 3
                stop = (stream + 1) * 3 if stream + 1 < visual_streams else output["video"].shape[1]
                output["video"][:, start:stop].masked_fill_(
                    channel_drop[:, stream, None, None, None, None], 0
                )
            output["visual_channel_drop"] = channel_drop
        if training and temporal_scale_jitter > 0:
            scales = torch.empty(len(output["label"]), device=device).uniform_(
                1 - temporal_scale_jitter, 1 + temporal_scale_jitter
            )
            output["video"] = temporal_resample(output["video"], scales, 2)
            output["skeleton"] = temporal_resample(output["skeleton"], scales, 1)
            output["imu"] = temporal_resample(output["imu"], scales, 1)
            if "radar" in output:
                output["radar"] = temporal_resample(output["radar"], scales, 1)
            output["temporal_scale"] = scales
    return output


def drop_one_available_modality(batch):
    output = batch.copy()
    availability = torch.cat([batch["visual_mask"].bool(), batch["sensor_mask"].bool()], dim=1)
    eligible = availability.sum(dim=1) >= 2
    scores = torch.rand(availability.shape, device=availability.device).masked_fill(~availability, 2)
    selected = scores.argmin(dim=1)
    dropped = torch.zeros_like(availability)
    dropped[torch.arange(len(dropped), device=dropped.device), selected] = eligible

    output["video"] = batch["video"].clone()
    output["visual_mask"] = batch["visual_mask"].clone() * (~dropped[:, :3])
    visual_streams = batch["visual_mask"].shape[1]
    for stream in range(visual_streams):
        start = stream * 3
        stop = (stream + 1) * 3 if stream + 1 < visual_streams else output["video"].shape[1]
        output["video"][:, start:stop].masked_fill_(
            dropped[:, stream, None, None, None, None], 0
        )

    output["skeleton"] = batch["skeleton"].clone()
    output["imu"] = batch["imu"].clone()
    output["sensor_mask"] = batch["sensor_mask"].clone() * (~dropped[:, 3:])
    output["skeleton"].masked_fill_(
        dropped[:, 3].reshape(len(dropped), *([1] * (output["skeleton"].ndim - 1))), 0
    )
    output["imu"].masked_fill_(
        dropped[:, 4].reshape(len(dropped), *([1] * (output["imu"].ndim - 1))), 0
    )
    output["distilled_modality_drop"] = dropped
    return output


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, coefficient):
        ctx.coefficient = coefficient
        return values.view_as(values)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.coefficient * gradient, None


class CrossUserContrastiveQueue:
    def __init__(
        self, dimension, maximum_size, temperature, device, class_cap=0, class_user_cap=0
    ):
        self.maximum_size = maximum_size
        self.temperature = temperature
        self.class_cap = class_cap
        self.class_user_cap = class_user_cap
        self.features = torch.empty(0, dimension, device=device)
        self.labels = torch.empty(0, dtype=torch.long, device=device)
        self.users = torch.empty(0, dtype=torch.long, device=device)

    def loss(self, features, labels, users):
        anchors = functional.normalize(features.float(), dim=1)
        candidates = torch.cat([anchors, self.features], dim=0)
        candidate_labels = torch.cat([labels, self.labels], dim=0)
        candidate_users = torch.cat([users, self.users], dim=0)
        same_label = labels[:, None] == candidate_labels[None, :]
        same_user = users[:, None] == candidate_users[None, :]
        positives = same_label & ~same_user
        valid = ~same_label | positives
        current = torch.arange(len(labels), device=labels.device)
        valid[current, current] = False
        positives[current, current] = False
        active = positives.any(dim=1)
        if not active.any():
            return anchors.sum() * 0, 0
        logits = anchors[active] @ candidates.t() / self.temperature
        active_valid = valid[active]
        active_positives = positives[active]
        logits = logits.masked_fill(~active_valid, -torch.inf)
        log_probabilities = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        positive_log_probabilities = torch.where(active_positives, log_probabilities, 0)
        losses = -positive_log_probabilities.sum(dim=1) / active_positives.sum(dim=1)
        return losses.mean(), int(active.sum())

    @torch.no_grad()
    def update(self, features, labels, users):
        if self.maximum_size == 0:
            return
        normalized = functional.normalize(features.float(), dim=1).detach()
        for feature, label, user in zip(normalized, labels.detach(), users.detach()):
            if self.class_user_cap:
                pair = (self.labels == label) & (self.users == user)
                if int(pair.sum()) >= self.class_user_cap:
                    self._remove(int(pair.nonzero()[0]))
            if self.class_cap:
                same_class = self.labels == label
                if int(same_class.sum()) >= self.class_cap:
                    self._remove(int(same_class.nonzero()[0]))
            if len(self.labels) >= self.maximum_size:
                self._remove(0)
            self.features = torch.cat([self.features, feature.unsqueeze(0)])
            self.labels = torch.cat([self.labels, label.unsqueeze(0)])
            self.users = torch.cat([self.users, user.unsqueeze(0)])

    def _remove(self, position):
        keep = torch.ones(len(self.labels), dtype=torch.bool, device=self.labels.device)
        keep[position] = False
        self.features = self.features[keep]
        self.labels = self.labels[keep]
        self.users = self.users[keep]


class ConditionalSubjectHead(nn.Module):
    def __init__(self, dimension, classes, users):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(classes, users, dimension))
        self.bias = nn.Parameter(torch.zeros(classes, users))
        nn.init.trunc_normal_(self.weight, std=0.02)

    def forward(self, features, labels):
        weights = self.weight[labels]
        return torch.einsum("bd,bud->bu", features, weights) + self.bias[labels]


def forward(model, batch):
    return model(
        batch["video"],
        skeleton=batch.get("skeleton"),
        imu=batch.get("imu"),
        radar=batch.get("radar"),
        sensor_mask=batch.get("sensor_mask"),
        visual_mask=batch.get("visual_mask"),
    )


def mixed_cross_entropy(logits, batch, weight=None, label_smoothing=0.0, reduction="mean"):
    primary = functional.cross_entropy(
        logits, batch["label"], weight=weight, label_smoothing=label_smoothing, reduction=reduction
    )
    if "mix_label" not in batch:
        return primary
    secondary = functional.cross_entropy(
        logits, batch["mix_label"], weight=weight, label_smoothing=label_smoothing, reduction=reduction
    )
    return batch["mix_lambda"] * primary + (1 - batch["mix_lambda"]) * secondary


def group_dro_cross_entropy(
    logits,
    batch,
    index_to_user,
    group_weights,
    eta,
    weight=None,
    label_smoothing=0.0,
):
    primary = functional.cross_entropy(
        logits,
        batch["label"],
        weight=weight,
        label_smoothing=label_smoothing,
        reduction="none",
    )
    primary_users = index_to_user[batch["index"].to(logits.device, non_blocking=True)]
    group_loss_sum = torch.zeros_like(group_weights)
    group_mass = torch.zeros_like(group_weights)
    primary_mass = torch.ones_like(primary)
    if "mix_label" in batch:
        secondary = functional.cross_entropy(
            logits,
            batch["mix_label"],
            weight=weight,
            label_smoothing=label_smoothing,
            reduction="none",
        )
        secondary_users = index_to_user[
            batch["mix_index"].to(logits.device, non_blocking=True)
        ]
        primary_mass = primary_mass * batch["mix_lambda"]
        secondary_mass = torch.ones_like(secondary) * (1 - batch["mix_lambda"])
        group_loss_sum.scatter_add_(0, secondary_users, secondary * secondary_mass)
        group_mass.scatter_add_(0, secondary_users, secondary_mass)
    group_loss_sum.scatter_add_(0, primary_users, primary * primary_mass)
    group_mass.scatter_add_(0, primary_users, primary_mass)
    observed = group_mass > 0
    group_losses = group_loss_sum / group_mass.clamp_min(1e-8)
    with torch.no_grad():
        group_weights[observed] *= torch.exp(eta * group_losses[observed].detach()).clamp_max(10)
        group_weights /= group_weights.sum().clamp_min(1e-8)
    normalized_weights = group_weights[observed] / group_weights[observed].sum().clamp_min(1e-8)
    return (normalized_weights * group_losses[observed]).sum(), group_losses.detach(), observed


def hierarchical_cross_entropy(logits, batch):
    category_logits = torch.stack(
        [
            torch.logsumexp(logits[:, start:stop], dim=1)
            for start, stop in ((0, 6), (6, 12), (12, 17), (17, 23), (23, 28), (28, 37), (37, 40))
        ],
        dim=1,
    )
    label_to_category = torch.tensor(
        [0] * 6 + [1] * 6 + [2] * 5 + [3] * 6 + [4] * 5 + [5] * 9 + [6] * 3,
        dtype=torch.long,
        device=logits.device,
    )
    primary = functional.cross_entropy(category_logits, label_to_category[batch["label"]])
    if "mix_label" not in batch:
        return primary
    secondary = functional.cross_entropy(category_logits, label_to_category[batch["mix_label"]])
    return batch["mix_lambda"] * primary + (1 - batch["mix_lambda"]) * secondary


def mixed_stream_cross_entropy(logits, batch, weight=None, label_smoothing=0.0):
    batch_size, streams, classes = logits.shape
    flattened = logits.reshape(batch_size * streams, classes)
    primary_labels = batch["label"].unsqueeze(1).expand(-1, streams).reshape(-1)
    primary = functional.cross_entropy(
        flattened,
        primary_labels,
        weight=weight,
        label_smoothing=label_smoothing,
        reduction="none",
    ).reshape(batch_size, streams)
    if "mix_label" not in batch:
        return primary
    secondary_labels = batch["mix_label"].unsqueeze(1).expand(-1, streams).reshape(-1)
    secondary = functional.cross_entropy(
        flattened,
        secondary_labels,
        weight=weight,
        label_smoothing=label_smoothing,
        reduction="none",
    ).reshape(batch_size, streams)
    return batch["mix_lambda"] * primary + (1 - batch["mix_lambda"]) * secondary


def mixup_batch(batch, alpha):
    if alpha <= 0:
        return
    coefficient = float(np.random.beta(alpha, alpha))
    permutation = torch.randperm(len(batch["label"]), device=batch["label"].device)
    for name in ("video", "skeleton", "imu", "radar"):
        if name in batch:
            batch[name] = coefficient * batch[name] + (1 - coefficient) * batch[name][permutation]
    batch["visual_mask"] = torch.maximum(batch["visual_mask"], batch["visual_mask"][permutation])
    if "sensor_mask" in batch:
        batch["sensor_mask"] = torch.maximum(batch["sensor_mask"], batch["sensor_mask"][permutation])
    batch["mix_label"] = batch["label"][permutation]
    batch["mix_index"] = batch["index"][permutation.to(batch["index"].device)]
    batch["mix_lambda"] = coefficient


@torch.no_grad()
def evaluate(model, loader, device, args):
    model.eval()
    probabilities = []
    labels = []
    indices = []
    loss_sum = 0
    seen = 0
    for batch_number, raw_batch in enumerate(loader):
        batch = prepare_batch(raw_batch, device, args.mode in ("multimodal", "shared_multimodal"))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = forward(model, batch)
            loss = functional.cross_entropy(logits, batch["label"])
        count = len(batch["label"])
        loss_sum += float(loss.detach()) * count
        seen += count
        probabilities.append(logits.float().softmax(dim=1).cpu())
        labels.append(batch["label"].cpu())
        indices.append(batch["index"].cpu())
        if args.limit_valid_batches and batch_number + 1 >= args.limit_valid_batches:
            break
    probabilities = torch.cat(probabilities).numpy()
    labels = torch.cat(labels).numpy()
    indices = torch.cat(indices).numpy()
    records = get_records()
    users = np.asarray([records[int(index)]["user"] for index in indices], dtype=np.int64)
    correct = probabilities.argmax(axis=1) == labels
    user_accuracies = np.asarray([correct[users == user].mean() for user in np.unique(users)])
    metrics = {
        "loss": loss_sum / seen,
        "accuracy": float(correct.mean()),
        "user_macro_accuracy": float(user_accuracies.mean()),
        "user_min_accuracy": float(user_accuracies.min()),
        "user_accuracy_std": float(user_accuracies.std()),
    }
    if (
        args.selection_labels
        or args.selection_environment >= 0
        or args.selection_minimum_modalities > 0
    ):
        selected = np.ones(len(labels), dtype=bool)
        if args.selection_labels:
            selected &= np.isin(labels, args.selection_labels)
        if args.selection_environment >= 0:
            selected &= (users > 15) == bool(args.selection_environment)
        if args.selection_minimum_modalities > 0:
            selected &= np.asarray(
                [
                    len(records[int(index)]["modalities"])
                    >= args.selection_minimum_modalities
                    for index in indices
                ],
                dtype=bool,
            )
        if not selected.any():
            raise ValueError("Validation has no rows for the selection subset")
        metrics["selection_accuracy"] = float(correct[selected].mean())
        metrics["selection_rows"] = int(selected.sum())
    return metrics, {"probabilities": probabilities, "labels": labels, "indices": indices, "users": users}


def main():
    args = parse_args()
    args.training_labels = sorted(
        {int(value) for value in args.training_labels.split(",") if value}
    )
    args.selection_labels = sorted(
        {int(value) for value in args.selection_labels.split(",") if value}
    )
    args.focus_labels = sorted(
        {int(value) for value in args.focus_labels.split(",") if value}
    )
    args.validation_users = sorted(
        {int(value) for value in args.validation_users.split(",") if value}
    )
    if args.training_labels and not args.selection_labels:
        args.selection_labels = list(args.training_labels)
    if any(
        label < 0 or label >= 40
        for label in args.training_labels + args.selection_labels + args.focus_labels
    ):
        raise ValueError("Training, selection, and focus labels must be in [0, 39]")
    if args.focus_weight < 1 or bool(args.focus_labels) != (args.focus_weight > 1):
        raise ValueError("Use focus labels exactly when focus weight is greater than one")
    if not torch.cuda.is_available():
        raise RuntimeError("R(2+1)D training requires CUDA")
    seed_everything(args.seed + args.fold)
    device = torch.device("cuda")
    records = get_records()
    if args.validation_users:
        if args.fold < 0:
            raise ValueError("Custom validation users require a validation fold")
        known_users = {
            record["user"] for record in records if record["split"] == "train"
        }
        if not set(args.validation_users).issubset(known_users):
            raise ValueError("Custom validation users are absent from training metadata")
        train_indices = [
            record["index"]
            for record in records
            if record["split"] == "train"
            and record["user"] not in args.validation_users
        ]
        valid_indices = [
            record["index"]
            for record in records
            if record["split"] == "train"
            and record["user"] in args.validation_users
        ]
        if {
            records[index]["label"] for index in train_indices
        } != set(range(40)):
            raise ValueError("Custom training split is missing classes")
    else:
        train_indices, valid_indices = get_indices(records, args.fold)
    if args.training_environment >= 0:
        train_indices = [
            index
            for index in train_indices
            if int(records[index]["user"] > 15) == args.training_environment
        ]
        if args.selection_environment < 0:
            args.selection_environment = args.training_environment
    if not 0 <= args.training_minimum_modalities <= 6:
        raise ValueError("Training minimum modalities must be in [0, 6]")
    if not 0 <= args.selection_minimum_modalities <= 6:
        raise ValueError("Selection minimum modalities must be in [0, 6]")
    if args.training_minimum_modalities > 0:
        train_indices = [
            index
            for index in train_indices
            if len(records[index]["modalities"])
            >= args.training_minimum_modalities
        ]
    if args.training_labels:
        train_indices = [
            index for index in train_indices if records[index]["label"] in args.training_labels
        ]
        if len(train_indices) < args.batch_size:
            raise ValueError("Training-label filter leaves fewer than one batch")
    if not 0 <= args.class_balance_power <= 1:
        raise ValueError("Class balance power must be in [0, 1]")
    if args.class_balanced_sampling and args.class_user_balanced_sampling:
        raise ValueError("Choose only one balanced sampling strategy")
    if (
        args.group_dro_eta < 0
        or args.hierarchical_loss_weight < 0
        or args.balanced_softmax_temperature < 0
    ):
        raise ValueError("Robust and long-tail loss controls must be nonnegative")
    if not 0 <= args.visual_channel_dropout < 1 or not 0 <= args.structured_modality_dropout < 1:
        raise ValueError("Modality dropout probabilities must be in [0, 1)")
    if args.masked_modality_distillation_weight < 0:
        raise ValueError("Masked-modality distillation weight must be nonnegative")
    if args.masked_modality_distillation_temperature <= 0:
        raise ValueError("Masked-modality distillation temperature must be positive")
    if (
        args.cross_user_contrastive_weight < 0
        or args.cross_environment_contrastive_weight < 0
        or args.subject_adversarial_weight < 0
    ):
        raise ValueError("Generalization loss weights must be nonnegative")
    if args.backbone_bn_momentum != -1 and not 0 <= args.backbone_bn_momentum <= 1:
        raise ValueError("Backbone BN momentum must be -1 or in [0, 1]")
    if not 0 <= args.temporal_scale_jitter < 1:
        raise ValueError("Temporal scale jitter must be in [0, 1)")
    if not 0 <= args.blend_loss_weight <= 1 or not 0 <= args.blend_base_weight < 1:
        raise ValueError("Blend loss and base weights must be in valid probability ranges")
    if args.blend_r2_temperature <= 0:
        raise ValueError("Blend R(2+1)D temperature must be positive")
    if (args.blend_teacher is None) != (args.blend_loss_weight == 0):
        raise ValueError("Supply --blend-teacher exactly when --blend-loss-weight is positive")
    if args.use_radar and args.mode not in ("multimodal", "shared_multimodal"):
        raise ValueError("Radar requires a multimodal model")
    if args.visual_strategy != "sequential" and args.mode != "shared_multimodal":
        raise ValueError("Alternative visual strategies require shared multimodal mode")
    if args.use_temporal_embedding and args.use_phase_temporal_embedding:
        raise ValueError("Index and phase temporal embeddings are mutually exclusive")
    if args.per_visual_auxiliary and args.mode != "shared_multimodal":
        raise ValueError("Per-stream visual supervision requires shared multimodal mode")
    if args.visual_multiscale_tokens and args.mode != "shared_multimodal":
        raise ValueError("Visual multiscale tokens require shared multimodal mode")
    if args.environment_auxiliary_weight < 0:
        raise ValueError("Environment auxiliary weight must be nonnegative")
    if args.init_skeleton is not None and args.mode not in ("multimodal", "shared_multimodal"):
        raise ValueError("Skeleton initialization requires a multimodal model")
    if args.init_model is not None and (args.init_visual is not None or args.init_skeleton is not None):
        raise ValueError("Full-model initialization cannot be combined with encoder initialization")
    if args.canonical_pose and args.mode not in ("multimodal", "shared_multimodal"):
        raise ValueError("Canonical pose requires a multimodal model")
    if not 0 <= args.photometric_jitter <= 0.5:
        raise ValueError("Photometric jitter must be in [0, 0.5]")
    if not 0 <= args.photometric_gamma <= 1:
        raise ValueError("Photometric gamma range must be in [0, 1]")
    if args.mixup_alpha < 0:
        raise ValueError("Mixup alpha must be nonnegative")
    if not 0 <= args.mixstyle_probability <= 1 or args.mixstyle_alpha <= 0:
        raise ValueError("MixStyle probability and alpha must be positive valid values")
    if not 1 <= args.fusion_depth <= 8:
        raise ValueError("Fusion depth must be in [1, 8]")
    if not 0 <= args.horizontal_flip_probability <= 1:
        raise ValueError("Horizontal flip probability must be in [0, 1]")
    if (
        args.contrastive_temperature <= 0
        or args.contrastive_queue_size < 0
        or args.contrastive_class_cap < 0
        or args.contrastive_class_user_cap < 0
        or args.contrastive_warmup_items < 0
    ):
        raise ValueError("Invalid contrastive queue configuration")
    enabled_methods = sum(
        value > 0
        for value in (
            args.visual_channel_dropout,
            args.structured_modality_dropout,
            args.masked_modality_distillation_weight,
            args.cross_user_contrastive_weight,
            args.cross_environment_contrastive_weight,
            args.subject_adversarial_weight,
            float(args.freeze_backbone_bn),
            float(args.backbone_bn_momentum >= 0),
            float(args.class_balance_power > 0),
            float(args.class_balanced_sampling),
            float(args.class_user_balanced_sampling),
            args.group_dro_eta,
            args.hierarchical_loss_weight,
            args.balanced_softmax_temperature,
            args.environment_auxiliary_weight,
            args.temporal_scale_jitter,
            args.blend_loss_weight,
            args.photometric_jitter,
            args.photometric_gamma,
            args.mixup_alpha,
            args.mixstyle_probability,
            float(args.skeleton_scale_randomization),
        )
    )
    if enabled_methods > 1 and not args.allow_method_combination:
        raise ValueError("Enable at most one generalization method per confirmatory run")
    if (
        args.visual_channel_dropout > 0
        or args.structured_modality_dropout > 0
        or args.masked_modality_distillation_weight > 0
        or args.cross_user_contrastive_weight > 0
        or args.cross_environment_contrastive_weight > 0
        or args.subject_adversarial_weight > 0
        or args.freeze_backbone_bn
        or args.backbone_bn_momentum >= 0
        or args.class_balance_power > 0
        or args.class_balanced_sampling
        or args.class_user_balanced_sampling
        or args.group_dro_eta > 0
        or args.hierarchical_loss_weight > 0
        or args.balanced_softmax_temperature > 0
        or args.environment_auxiliary_weight > 0
        or args.temporal_scale_jitter > 0
        or args.blend_loss_weight > 0
        or args.photometric_jitter > 0
        or args.mixup_alpha > 0
        or args.mixstyle_probability > 0
        or args.skeleton_scale_randomization
    ) and args.mode not in ("multimodal", "shared_multimodal"):
        raise ValueError("Generalization methods require a multimodal model")
    class_counts = np.bincount([records[index]["label"] for index in train_indices], minlength=40).clip(1)
    class_weights = (class_counts.mean() / class_counts) ** args.class_balance_power
    if args.focus_labels:
        class_weights[args.focus_labels] *= args.focus_weight
    class_weights = torch.tensor(class_weights / class_weights.mean(), device=device, dtype=torch.float32)
    balanced_softmax_adjustment = args.balanced_softmax_temperature * torch.tensor(
        np.log(class_counts / class_counts.sum()), device=device, dtype=torch.float32
    )
    blend_teacher = None
    blend_teacher_sha256 = None
    if args.blend_teacher is not None:
        teacher_file = np.load(args.blend_teacher)
        teacher_indices = teacher_file["indices"].astype(np.int64)
        teacher_probabilities = teacher_file["probabilities"].astype(np.float32)
        if teacher_probabilities.shape != (len(teacher_indices), 40):
            raise ValueError("Blend teacher probabilities must have shape [samples, 40]")
        if "labels" in teacher_file:
            teacher_labels = teacher_file["labels"].astype(np.int64)
            expected_labels = np.asarray([records[int(index)]["label"] for index in teacher_indices])
            if not np.array_equal(teacher_labels, expected_labels):
                raise ValueError("Blend teacher labels do not match metadata")
        if not set(train_indices).issubset(set(teacher_indices.tolist())):
            raise ValueError("Blend teacher does not cover every training sample")
        blend_teacher = torch.zeros(len(records), 40, dtype=torch.float32, device=device)
        blend_teacher[torch.from_numpy(teacher_indices).to(device)] = torch.from_numpy(
            teacher_probabilities
        ).to(device)
        blend_teacher_sha256 = hashlib.sha256(args.blend_teacher.read_bytes()).hexdigest()
    include_sensors = args.mode in ("multimodal", "shared_multimodal")
    loader_options = {
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
    train_generator = torch.Generator().manual_seed(args.seed + args.fold)
    train_sampler = None
    if args.class_balanced_sampling or args.class_user_balanced_sampling:
        if args.class_user_balanced_sampling:
            class_users = {
                label: {
                    records[index]["user"]
                    for index in train_indices
                    if records[index]["label"] == label
                }
                for label in range(40)
            }
            class_user_counts = {}
            for index in train_indices:
                key = (records[index]["label"], records[index]["user"])
                class_user_counts[key] = class_user_counts.get(key, 0) + 1
            sample_weights = torch.tensor(
                [
                    1
                    / (
                        len(class_users[records[index]["label"]])
                        * class_user_counts[
                            (records[index]["label"], records[index]["user"])
                        ]
                    )
                    for index in train_indices
                ],
                dtype=torch.double,
            )
        else:
            sample_weights = torch.tensor(
                [1 / class_counts[records[index]["label"]] for index in train_indices],
                dtype=torch.double,
            )
        train_sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(train_indices),
            replacement=True,
            generator=train_generator,
        )
    train_loader = DataLoader(
        R2Plus1DDataset(
            train_indices, args.cache_dir, training=True, include_sensors=include_sensors,
            include_radar=args.use_radar, frames=args.frames, image_size=args.image_size,
            radar_representation=args.radar_representation,
            imu_representation=args.imu_representation,
            photometric_jitter=args.photometric_jitter, canonical_pose=args.canonical_pose,
            photometric_gamma=args.photometric_gamma,
            horizontal_flip_probability=args.horizontal_flip_probability,
            synchronized_horizontal_flip=args.synchronized_horizontal_flip,
            skeleton_scale_randomization=args.skeleton_scale_randomization,
        ),
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=True,
        generator=train_generator,
        **loader_options,
    )
    valid_loader = (
        DataLoader(
            R2Plus1DDataset(
                valid_indices, args.cache_dir, training=False, include_sensors=include_sensors,
                include_radar=args.use_radar, frames=args.frames, image_size=args.image_size,
                radar_representation=args.radar_representation,
                imu_representation=args.imu_representation,
                canonical_pose=args.canonical_pose,
                synchronized_horizontal_flip=args.synchronized_horizontal_flip,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            **loader_options,
        )
        if valid_indices else None
    )
    weights = args.weights if args.initialization == "kinetics" else None
    input_channels = train_loader.dataset.channels
    model = R2Plus1DClassifier(
        args.mode,
        weights=weights,
        dropout=args.dropout,
        input_channels=input_channels,
        mask_joint_visual=args.structured_modality_dropout > 0,
        use_radar=args.use_radar,
        skeleton_representation=args.skeleton_representation,
        mixstyle_probability=args.mixstyle_probability,
        mixstyle_alpha=args.mixstyle_alpha,
        fusion_depth=args.fusion_depth,
        use_temporal_embedding=args.use_temporal_embedding,
        fusion_output=args.fusion_output,
        visual_motion_channels=args.visual_motion_channels,
        visual_strategy=args.visual_strategy,
        use_phase_temporal_embedding=args.use_phase_temporal_embedding,
        per_visual_auxiliary=args.per_visual_auxiliary,
        environment_moe=args.environment_auxiliary_weight > 0,
        visual_multiscale_tokens=args.visual_multiscale_tokens,
        normalize_modality_tokens=args.normalize_modality_tokens,
        skeleton_encoder=args.skeleton_encoder,
        radar_representation=args.radar_representation,
        imu_inputs={"world": 85, "relative": 95, "combined": 180, "world_magnetic": 100, "world_official": 135}[
            args.imu_representation
        ],
        imu_statistics_residual=args.imu_statistics_residual,
        imu_statistics_logit_residual=args.imu_statistics_logit_residual,
        imu_device_attention_residual=args.imu_device_attention_residual,
        imu_statistics_inputs=50 if args.imu_representation == "world_official" else None,
    ).to(device)
    initialization_provenance = None
    if args.init_visual is not None:
        source = torch.load(args.init_visual, map_location="cpu", weights_only=False)
        if source["config"]["mode"] != "visual" or int(source["config"]["fold"]) != args.fold:
            raise ValueError("Visual initialization must come from the same validation fold")
        current = model.state_dict()
        compatible = {}
        adapted_tensors = []
        for name, value in source["model"].items():
            if not (name.startswith("video.") or name.startswith("video_head.")) or name not in current:
                continue
            if value.shape == current[name].shape:
                compatible[name] = value
            elif name == "video.stem.0.weight" and value.ndim == 5 and value.shape[0] == current[name].shape[0]:
                adapted = current[name].clone()
                copied = min(value.shape[1], adapted.shape[1])
                adapted[:, :copied] = value[:, :copied]
                if adapted.shape[1] > copied:
                    adapted[:, copied:] = value[:, :min(3, value.shape[1])].mean(dim=1, keepdim=True)
                compatible[name] = adapted
                adapted_tensors.append(name)
        expected = [name for name in current if name.startswith("video.") or name.startswith("video_head.")]
        if set(compatible) != set(expected):
            raise ValueError("Visual initialization is incomplete")
        model.load_state_dict(compatible, strict=False)
        initialization_provenance = {
            "path": str(args.init_visual),
            "sha256": hashlib.sha256(args.init_visual.read_bytes()).hexdigest(),
            "source_name": source["config"]["name"],
            "source_fold": source["config"]["fold"],
            "loaded_tensors": len(compatible),
            "adapted_tensors": adapted_tensors,
        }
    skeleton_initialization_provenance = None
    if args.init_skeleton is not None:
        source = torch.load(args.init_skeleton, map_location="cpu", weights_only=False)
        source_config = source["config"]
        if int(source_config["fold"]) != args.fold or source_config.get("modalities") != ["skeleton"]:
            raise ValueError("Skeleton initialization must come from the same fold's skeleton-only model")
        if source_config.get("skeleton_representation", "coordinates") != args.skeleton_representation:
            raise ValueError("Skeleton initialization representation does not match")
        if bool(source_config.get("canonical_pose", False)) != args.canonical_pose:
            raise ValueError("Skeleton initialization canonical-pose setting does not match")
        current = model.state_dict()
        compatible = {}
        for name, value in source["model"].items():
            destination_name = None
            if name.startswith("encoders.skeleton."):
                destination_name = "skeleton." + name.removeprefix("encoders.skeleton.")
            elif name.startswith("auxiliary.skeleton."):
                destination_name = "skeleton_head." + name.removeprefix("auxiliary.skeleton.")
            if destination_name is not None and destination_name in current:
                if value.shape != current[destination_name].shape:
                    raise ValueError(f"Skeleton initialization shape mismatch: {destination_name}")
                compatible[destination_name] = value
        expected = [
            name for name in current
            if name.startswith("skeleton.") or name.startswith("skeleton_head.")
        ]
        if set(compatible) != set(expected):
            raise ValueError("Skeleton initialization is incomplete")
        model.load_state_dict(compatible, strict=False)
        skeleton_initialization_provenance = {
            "path": str(args.init_skeleton),
            "sha256": hashlib.sha256(args.init_skeleton.read_bytes()).hexdigest(),
            "source_name": source_config["name"],
            "source_fold": source_config["fold"],
            "loaded_tensors": len(compatible),
        }
    model_initialization_provenance = None
    if args.init_model is not None:
        source = torch.load(args.init_model, map_location="cpu", weights_only=False)
        source_config = source["config"]
        if source_config["mode"] != args.mode or int(source_config["fold"]) != args.fold:
            raise ValueError("Full-model initialization must match mode and validation fold")
        source_skeleton_encoder = source_config.get("skeleton_encoder", "tcn")
        extends_radar = args.use_radar and not bool(source_config.get("use_radar", False))
        if extends_radar:
            current = model.state_dict()
            compatible = {}
            for name, value in source["model"].items():
                if name == "modality_embedding":
                    if value.shape[1] + 1 != current[name].shape[1]:
                        raise ValueError("Radar extension must add exactly one modality embedding")
                    expanded = current[name].clone()
                    expanded[:, :value.shape[1]].copy_(value)
                    compatible[name] = expanded
                elif name in current and value.shape == current[name].shape:
                    compatible[name] = value
                else:
                    raise ValueError(f"Radar extension source tensor is incompatible: {name}")
            incompatible = model.load_state_dict(compatible, strict=False)
            permitted_missing = {
                name for name in current
                if name.startswith("radar.") or name.startswith("radar_head.")
            }
            if set(incompatible.missing_keys) != permitted_missing or incompatible.unexpected_keys:
                raise ValueError("Full-model initialization is not a clean radar extension")
        elif args.skeleton_encoder == "residual_invariant" and source_skeleton_encoder == "tcn":
            incompatible = model.load_state_dict(source["model"], strict=False)
            permitted_missing = {
                name for name in model.state_dict()
                if name.startswith("skeleton.invariant.")
                or name.startswith("skeleton.view_projection.")
            }
            if set(incompatible.missing_keys) != permitted_missing or incompatible.unexpected_keys:
                raise ValueError("Full-model initialization is not a clean skeleton extension")
        elif (
            args.imu_representation != source_config.get("imu_representation", "world")
            or args.imu_statistics_residual
            or args.imu_statistics_logit_residual
            or args.imu_device_attention_residual
        ):
            current = model.state_dict()
            compatible = {}
            adapted_name = "imu.convolution.0.weight"
            for name, value in source["model"].items():
                if (
                    name == adapted_name
                    and args.imu_representation != source_config.get("imu_representation", "world")
                ):
                    adapted = current[name].clone()
                    adapted.zero_()
                    copied = min(value.shape[1], adapted.shape[1])
                    adapted[:, :copied].copy_(value[:, :copied])
                    compatible[name] = adapted
                elif name in current and value.shape == current[name].shape:
                    compatible[name] = value
                else:
                    raise ValueError(f"IMU extension source tensor is incompatible: {name}")
            incompatible = model.load_state_dict(compatible, strict=False)
            permitted_missing = (
                {
                    name for name in current
                    if name.startswith("imu_statistics.") or name == "imu_statistics_scale"
                }
                if args.imu_statistics_residual or args.imu_statistics_logit_residual else set()
            )
            if args.imu_statistics_logit_residual:
                permitted_missing |= {
                    name for name in current
                    if name.startswith("imu_statistics_head.")
                    or name == "imu_statistics_logit_scale"
                }
            if args.imu_device_attention_residual:
                permitted_missing |= {
                    name for name in current
                    if name.startswith("imu_device_")
                }
            if set(incompatible.missing_keys) != permitted_missing or incompatible.unexpected_keys:
                raise ValueError("Full-model initialization is not a clean IMU extension")
        else:
            model.load_state_dict(source["model"], strict=True)
        model_initialization_provenance = {
            "path": str(args.init_model),
            "sha256": hashlib.sha256(args.init_model.read_bytes()).hexdigest(),
            "source_name": source_config["name"],
            "source_fold": source_config["fold"],
            "loaded_tensors": len(source["model"]),
            "architecture_extension": (
                f"{args.radar_representation}_radar"
                if extends_radar
                else "residual_invariant_skeleton"
                if args.skeleton_encoder == "residual_invariant" and source_skeleton_encoder == "tcn"
                else f"{args.imu_representation}_imu"
                + (
                    "_statistics_residual" if args.imu_statistics_residual
                    else "_statistics_logit_residual" if args.imu_statistics_logit_residual
                    else "_device_attention_residual" if args.imu_device_attention_residual
                    else ""
                )
                if (
                    args.imu_representation != source_config.get("imu_representation", "world")
                    or args.imu_statistics_residual
                    or args.imu_statistics_logit_residual
                    or args.imu_device_attention_residual
                )
                else None
            ),
        }
    if args.freeze_visual:
        if (
            (args.init_visual is None and args.init_model is None)
            or args.mode not in ("multimodal", "shared_multimodal")
        ):
            raise ValueError(
                "Frozen visual training requires a visual or full-model initializer "
                "and multimodal mode"
            )
        for parameter in model.video.parameters():
            parameter.requires_grad_(False)
        for parameter in model.video_head.parameters():
            parameter.requires_grad_(False)
    if args.freeze_inherited:
        if args.init_model is None or not (
            args.imu_statistics_residual
            or args.imu_statistics_logit_residual
            or args.imu_device_attention_residual
        ):
            raise ValueError("Frozen inherited training requires a full-model initializer and IMU statistics residual")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(
                name.startswith("imu_statistics.") or name == "imu_statistics_scale"
                or name.startswith("imu_statistics_head.")
                or name == "imu_statistics_logit_scale"
                or name.startswith("imu_device_")
            )
    if args.backbone_bn_momentum >= 0:
        for module in model.video.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.momentum = args.backbone_bn_momentum
    capture_features = (
        args.cross_user_contrastive_weight > 0
        or args.cross_environment_contrastive_weight > 0
        or args.subject_adversarial_weight > 0
    )
    captured = {}
    feature_hook = None
    train_users = sorted({records[index]["user"] for index in train_indices})
    user_to_target = {user: target for target, user in enumerate(train_users)}
    index_to_user_target = torch.full((len(records),), -1, dtype=torch.long, device=device)
    index_to_environment = torch.full((len(records),), -1, dtype=torch.long, device=device)
    for index in train_indices:
        index_to_user_target[index] = user_to_target[records[index]["user"]]
    for record in records:
        if record["split"] == "train":
            index_to_environment[record["index"]] = int(record["user"] > 15)
    subject_head = None
    class_user_counts = torch.zeros(40, len(train_users), dtype=torch.float32, device=device)
    for index in train_indices:
        class_user_counts[records[index]["label"], user_to_target[records[index]["user"]]] += 1
    class_user_support = class_user_counts > 0
    class_user_sample_weights = class_user_counts.clamp_min(1).rsqrt()
    populated_weights = class_user_sample_weights[class_user_support]
    class_user_sample_weights = class_user_sample_weights / populated_weights.mean()
    group_dro_weights = torch.full(
        (len(train_users),), 1 / len(train_users), dtype=torch.float32, device=device
    )
    if args.subject_adversarial_weight > 0:
        subject_head = ConditionalSubjectHead(
            model.normalization.normalized_shape[0], 40, len(train_users)
        ).to(device)
    contrastive_queue = None
    if args.cross_user_contrastive_weight > 0:
        contrastive_queue = CrossUserContrastiveQueue(
            model.normalization.normalized_shape[0],
            args.contrastive_queue_size,
            args.contrastive_temperature,
            device,
            class_cap=args.contrastive_class_cap,
            class_user_cap=args.contrastive_class_user_cap,
        )
    environment_contrastive_queue = None
    if args.cross_environment_contrastive_weight > 0:
        environment_contrastive_queue = CrossUserContrastiveQueue(
            model.normalization.normalized_shape[0],
            args.contrastive_queue_size,
            args.contrastive_temperature,
            device,
            class_cap=args.contrastive_class_cap,
        )
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    if capture_features:
        feature_hook = model.normalization.register_forward_hook(
            lambda module, inputs, output: captured.__setitem__("features", output)
        )
    visual_adapter_ids = {
        id(parameter) for parameter in model.video.modality_affines.parameters()
    }
    backbone_parameters = [
        parameter for parameter in model.video.parameters()
        if parameter.requires_grad and id(parameter) not in visual_adapter_ids
    ]
    if args.init_visual is not None or args.init_model is not None:
        backbone_parameters += [parameter for parameter in model.video_head.parameters() if parameter.requires_grad]
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    head_parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in backbone_ids
    ]
    parameter_groups = []
    if backbone_parameters:
        parameter_groups.append({"params": backbone_parameters, "factor": args.backbone_lr_factor})
    if head_parameters:
        parameter_groups.append({"params": head_parameters, "factor": 1.0})
    if subject_head is not None:
        parameter_groups.append({"params": subject_head.parameters(), "factor": 1.0})
    optimizer = torch.optim.AdamW(parameter_groups, lr=args.lr, weight_decay=args.weight_decay)
    destination = args.output_root / "checkpoints" / args.name
    destination.mkdir(parents=True, exist_ok=True)
    source_files = [
        "scripts/train_r2plus1d.py",
        "src/r2plus1d_models.py",
        "src/r2plus1d_data.py",
        "src/scratch_models.py",
        "data/metadata/samples.json",
    ]
    config = vars(args).copy()
    config.update(
        {
            "cache_dir": str(args.cache_dir),
            "weights": str(args.weights),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "training_only_parameters": (
                sum(parameter.numel() for parameter in subject_head.parameters())
                if subject_head is not None else 0
            ),
            "visual_adapter_parameters": sum(
                parameter.numel() for parameter in model.video.modality_affines.parameters()
            ),
            "visual_adapter_lr_factor": 1.0,
            "input_channels": input_channels,
            "train_clips": len(train_indices),
            "valid_clips": len(valid_indices),
            "validation_users": (
                args.validation_users
                if args.validation_users
                else FOLDS[args.fold] if args.fold >= 0 else []
            ),
            "training_users": train_users,
            "mask_joint_visual": args.structured_modality_dropout > 0,
            "precision": "BF16 autocast; FP32 optimizer and EMA",
            "architecture": (
                "R(2+1)D-34 CNN"
                if args.mode == "visual"
                else f"shared R(2+1)D-34 CNN ({args.visual_strategy}) for Depth/Thermal/IR + {args.skeleton_encoder} skeleton CNN + IMU CNN/LSTM + {args.radar_representation + ' Radar CNN + ' if args.use_radar else ''}Transformer fusion"
                if args.mode == "shared_multimodal"
                else "R(2+1)D-34 CNN + temporal CNN/LSTM + Transformer fusion"
            ),
            "initialization_source": "IG65M to Kinetics-400" if weights else "random",
            "visual_initialization": initialization_provenance,
            "skeleton_initialization": skeleton_initialization_provenance,
            "model_initialization": model_initialization_provenance,
            "blend_teacher_sha256": blend_teacher_sha256,
            "source_sha256": {
                name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_files
            },
            "cache_manifest_sha256": hashlib.sha256((args.cache_dir / "manifest.json").read_bytes()).hexdigest(),
        }
    )
    if weights:
        config["weights_sha256"] = hashlib.sha256(weights.read_bytes()).hexdigest()
    if args.imu_representation in ("world_magnetic", "world_official"):
        for name in [
            "scripts/prepare_imu_magnetic.py",
            f"data/processed/imu_{args.imu_representation}.npy",
        ]:
            config["source_sha256"][name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    serializable = {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}
    (destination / "config.json").write_text(json.dumps(serializable, indent=2) + "\n")
    (destination / "split.json").write_text(json.dumps({"train": train_indices, "valid": valid_indices}) + "\n")

    import swanlab

    run = swanlab.init(
        project="cuhkx-small-multimodal",
        name=args.name,
        config=serializable,
        public=False,
        mode=args.swanlab_mode,
        log_dir=str(args.output_root / "swanlog"),
        description="Legal high-capacity candidate; user-isolated validation and training-only generalization objectives.",
    )
    print(json.dumps(serializable), flush=True)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = round(len(train_loader) * args.warmup_epochs)
    contrastive_warmup_steps = math.ceil(args.contrastive_warmup_items / args.batch_size)
    global_step = 0
    best_accuracy = -1.0
    best_epoch = 0
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.freeze_inherited:
            model.eval()
            if args.imu_device_attention_residual:
                model.imu_device_statistics.train()
                model.imu_device_attention.train()
                model.imu_device_normalization.train()
                model.imu_device_head.train()
            else:
                model.imu_statistics.train()
        elif args.freeze_visual:
            model.video.eval()
            model.video_head.eval()
        elif args.freeze_backbone_bn:
            for module in model.video.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()
        epoch_started = time.monotonic()
        loss_sum = 0
        correct = 0
        seen = 0
        contrastive_loss_sum = 0.0
        contrastive_active = 0
        environment_contrastive_loss_sum = 0.0
        environment_contrastive_active = 0
        subject_loss_sum = 0.0
        subject_correct = 0
        visual_channel_drops = 0
        visual_channel_opportunities = 0
        branch_names = ("video", "skeleton", "imu", "radar") if args.use_radar else (
            "video", "skeleton", "imu"
        )
        branch_drops = torch.zeros(len(branch_names), dtype=torch.long)
        branch_opportunities = torch.zeros(len(branch_names), dtype=torch.long)
        task_loss_sum = 0.0
        blend_loss_sum = 0.0
        environment_loss_sum = 0.0
        masked_modality_distillation_loss_sum = 0.0
        hierarchical_loss_sum = 0.0
        group_dro_active = torch.zeros(len(train_users), dtype=torch.bool, device=device)
        for batch_number, raw_batch in enumerate(train_loader):
            rate = learning_rate(global_step, total_steps, warmup_steps, args.lr)
            for group in optimizer.param_groups:
                group["lr"] = rate * group["factor"]
            batch = prepare_batch(
                raw_batch,
                device,
                include_sensors,
                training=True,
                visual_channel_dropout=args.visual_channel_dropout,
                structured_modality_dropout=args.structured_modality_dropout,
                temporal_scale_jitter=args.temporal_scale_jitter,
                default_modality_dropout=args.masked_modality_distillation_weight == 0,
            )
            mixup_batch(batch, args.mixup_alpha)
            teacher_batch = batch
            if args.masked_modality_distillation_weight > 0:
                batch = drop_one_available_modality(batch)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if args.masked_modality_distillation_weight > 0:
                    with torch.no_grad():
                        teacher_logits, _ = forward(ema, teacher_batch)
                logits, auxiliary = forward(model, batch)
                if args.group_dro_eta > 0:
                    task_cross_entropy, _, observed_groups = group_dro_cross_entropy(
                        logits + balanced_softmax_adjustment.unsqueeze(0),
                        batch,
                        index_to_user_target,
                        group_dro_weights,
                        args.group_dro_eta,
                        weight=class_weights,
                        label_smoothing=args.label_smoothing,
                    )
                    group_dro_active |= observed_groups
                elif args.balanced_softmax_temperature > 0:
                    task_cross_entropy = mixed_cross_entropy(
                        logits + balanced_softmax_adjustment.unsqueeze(0),
                        batch,
                        weight=class_weights,
                        label_smoothing=args.label_smoothing,
                    )
                else:
                    task_cross_entropy = mixed_cross_entropy(
                        logits, batch, weight=class_weights, label_smoothing=args.label_smoothing
                    )
                loss = task_cross_entropy
                blend_loss = logits.sum() * 0
                environment_loss = logits.sum() * 0
                masked_modality_distillation_loss = logits.sum() * 0
                hierarchical_loss = logits.sum() * 0
                if args.masked_modality_distillation_weight > 0:
                    temperature = args.masked_modality_distillation_temperature
                    masked_modality_distillation_loss = functional.kl_div(
                        functional.log_softmax(logits.float() / temperature, dim=1),
                        functional.softmax(teacher_logits.float() / temperature, dim=1),
                        reduction="batchmean",
                    ) * temperature ** 2
                    loss = loss + (
                        args.masked_modality_distillation_weight
                        * masked_modality_distillation_loss
                    )
                if blend_teacher is not None:
                    teacher = blend_teacher[batch["index"].to(device, non_blocking=True)]
                    r2_probabilities = (logits.float() / args.blend_r2_temperature).softmax(dim=1)
                    target = batch["label"].unsqueeze(1)
                    final_target_probability = (
                        args.blend_base_weight * teacher.gather(1, target)
                        + (1 - args.blend_base_weight) * r2_probabilities.gather(1, target)
                    )
                    blend_loss = -final_target_probability.clamp_min(1e-8).log().mean()
                    loss = (
                        (1 - args.blend_loss_weight) * task_cross_entropy
                        + args.blend_loss_weight * blend_loss
                    )
                if args.hierarchical_loss_weight > 0:
                    hierarchical_loss = hierarchical_cross_entropy(logits, batch)
                    loss = loss + args.hierarchical_loss_weight * hierarchical_loss
                if include_sensors:
                    if "environment" in auxiliary:
                        environment_targets = index_to_environment[
                            batch["index"].to(device, non_blocking=True)
                        ]
                        environment_loss = functional.cross_entropy(
                            auxiliary["environment"], environment_targets
                        )
                        if "mix_label" in batch:
                            mixed_environment_targets = index_to_environment[
                                batch["mix_index"].to(device, non_blocking=True)
                            ]
                            secondary_environment_loss = functional.cross_entropy(
                                auxiliary["environment"], mixed_environment_targets
                            )
                            environment_loss = (
                                batch["mix_lambda"] * environment_loss
                                + (1 - batch["mix_lambda"]) * secondary_environment_loss
                            )
                        loss = loss + args.environment_auxiliary_weight * environment_loss
                    classification_auxiliary_count = len(auxiliary) - int("environment" in auxiliary)
                    for name, branch_logits in auxiliary.items():
                        if name == "environment":
                            continue
                        if name == "video_streams":
                            branch_loss = mixed_stream_cross_entropy(
                                branch_logits,
                                batch,
                                weight=class_weights,
                                label_smoothing=args.label_smoothing,
                            )
                            mask = batch["visual_mask"]
                            loss = loss + args.auxiliary_weight / classification_auxiliary_count * (
                                branch_loss * mask
                            ).sum() / mask.sum().clamp_min(1)
                            continue
                        branch_loss = mixed_cross_entropy(
                            branch_logits, batch, weight=class_weights,
                            reduction="none", label_smoothing=args.label_smoothing
                        )
                        if name == "video":
                            mask = (
                                batch["visual_mask"].bool().any(dim=1).float()
                                if args.structured_modality_dropout > 0
                                else torch.ones_like(batch["label"], dtype=torch.float32)
                            )
                        else:
                            mask = batch["sensor_mask"][:, {"skeleton": 0, "imu": 1, "radar": 2}[name]]
                        loss = loss + args.auxiliary_weight / classification_auxiliary_count * (
                            branch_loss * mask
                        ).sum() / mask.sum().clamp_min(1)
                if capture_features:
                    features = captured.pop("features")
                    users = index_to_user_target[batch["index"].to(device, non_blocking=True)]
                    feature_environments = index_to_environment[
                        batch["index"].to(device, non_blocking=True)
                    ]
                    feature_labels = batch["label"]
                    if "mix_label" in batch and batch["mix_lambda"] < 0.5:
                        feature_labels = batch["mix_label"]
                        users = index_to_user_target[batch["mix_index"].to(device, non_blocking=True)]
                        feature_environments = index_to_environment[
                            batch["mix_index"].to(device, non_blocking=True)
                        ]
                if contrastive_queue is not None:
                    if global_step >= contrastive_warmup_steps:
                        contrastive_loss, active_anchors = contrastive_queue.loss(
                            features, feature_labels, users
                        )
                        contrastive_ramp = min(
                            (global_step - contrastive_warmup_steps + 1) / max(len(train_loader), 1), 1
                        )
                        contrastive_weight = args.cross_user_contrastive_weight * contrastive_ramp
                        loss = loss + contrastive_weight * contrastive_loss
                    else:
                        contrastive_loss = features.sum() * 0
                        active_anchors = 0
                        contrastive_weight = 0.0
                if environment_contrastive_queue is not None:
                    if global_step >= contrastive_warmup_steps:
                        environment_contrastive_loss, environment_active_anchors = (
                            environment_contrastive_queue.loss(
                                features, feature_labels, feature_environments
                            )
                        )
                        environment_contrastive_ramp = min(
                            (global_step - contrastive_warmup_steps + 1)
                            / max(len(train_loader), 1),
                            1,
                        )
                        environment_contrastive_weight = (
                            args.cross_environment_contrastive_weight
                            * environment_contrastive_ramp
                        )
                        loss = loss + (
                            environment_contrastive_weight
                            * environment_contrastive_loss
                        )
                    else:
                        environment_contrastive_loss = features.sum() * 0
                        environment_active_anchors = 0
                        environment_contrastive_weight = 0.0
                if subject_head is not None:
                    adversarial_steps = max(total_steps - len(train_loader), 1)
                    adversarial_progress = max(global_step - len(train_loader), 0) / adversarial_steps
                    adversarial_coefficient = 0.0 if global_step < len(train_loader) else (
                        args.subject_adversarial_weight
                        * (2 / (1 + math.exp(-10 * adversarial_progress)) - 1)
                    )
                    subject_logits = subject_head(
                        GradientReversal.apply(features.float(), adversarial_coefficient),
                        feature_labels,
                    )
                    subject_logits = subject_logits.masked_fill(
                        ~class_user_support[feature_labels], -1e4
                    )
                    subject_losses = functional.cross_entropy(
                        subject_logits, users, reduction="none"
                    )
                    subject_weights = class_user_sample_weights[feature_labels, users]
                    subject_loss = (subject_losses * subject_weights).sum() / subject_weights.sum()
                    loss = loss + subject_loss
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss")
            loss.backward()
            parameters_to_clip = list(model.parameters())
            if subject_head is not None:
                parameters_to_clip += list(subject_head.parameters())
            torch.nn.utils.clip_grad_norm_(parameters_to_clip, 5)
            optimizer.step()
            if contrastive_queue is not None:
                contrastive_queue.update(features, feature_labels, users)
            if environment_contrastive_queue is not None:
                environment_contrastive_queue.update(
                    features, feature_labels, feature_environments
                )
            with torch.no_grad():
                decay = min(args.ema_decay, 1 - 1 / (global_step + 1))
                for target, source in zip(ema.parameters(), model.parameters()):
                    target.lerp_(source, 1 - decay)
                for target, source in zip(ema.buffers(), model.buffers()):
                    target.copy_(source)
            count = len(batch["label"])
            loss_sum += float(loss.detach()) * count
            task_loss_sum += float(mixed_cross_entropy(logits.detach(), batch)) * count
            blend_loss_sum += float(blend_loss.detach()) * count
            environment_loss_sum += float(environment_loss.detach()) * count
            masked_modality_distillation_loss_sum += (
                float(masked_modality_distillation_loss.detach()) * count
            )
            hierarchical_loss_sum += float(hierarchical_loss.detach()) * count
            correct += int((logits.argmax(dim=1) == batch["label"]).sum())
            seen += count
            if "visual_channel_drop" in batch:
                visual_channel_drops += int(batch["visual_channel_drop"].sum())
                visual_channel_opportunities += int((raw_batch["visual_mask"].sum(dim=1) >= 2).sum())
            if "branch_drop" in batch:
                branch_drops += batch["branch_drop"].sum(dim=0).cpu()
                natural_visual = raw_batch["visual_mask"].bool().any(dim=1, keepdim=True)
                natural_branches = torch.cat([natural_visual, raw_batch["sensor_mask"].bool()], dim=1)
                branch_opportunities += natural_branches.sum(dim=0)
            if contrastive_queue is not None:
                contrastive_loss_sum += float(contrastive_loss.detach()) * active_anchors
                contrastive_active += active_anchors
            if environment_contrastive_queue is not None:
                environment_contrastive_loss_sum += (
                    float(environment_contrastive_loss.detach())
                    * environment_active_anchors
                )
                environment_contrastive_active += environment_active_anchors
            if subject_head is not None:
                subject_loss_sum += float(subject_loss.detach()) * count
                subject_correct += int((subject_logits.argmax(dim=1) == users).sum())
            global_step += 1
            if args.limit_train_batches and batch_number + 1 >= args.limit_train_batches:
                break
        metrics = {
            "train/loss": loss_sum / seen,
            "train/task_cross_entropy": task_loss_sum / seen,
            "train/accuracy": correct / seen,
            "train/lr": rate,
            "system/epoch_seconds": time.monotonic() - epoch_started,
            "system/peak_cuda_mb": torch.cuda.max_memory_allocated() / 1e6,
        }
        if contrastive_queue is not None:
            metrics.update({
                "train/cross_user_contrastive_loss": contrastive_loss_sum / seen,
                "train/contrastive_active_fraction": contrastive_active / seen,
                "train/contrastive_queue_size": len(contrastive_queue.labels),
                "train/contrastive_weight": contrastive_weight,
            })
            metrics["train/cross_user_contrastive_loss"] = (
                contrastive_loss_sum / max(contrastive_active, 1)
            )
        if environment_contrastive_queue is not None:
            metrics.update(
                {
                    "train/cross_environment_contrastive_loss": (
                        environment_contrastive_loss_sum
                        / max(environment_contrastive_active, 1)
                    ),
                    "train/environment_contrastive_active_fraction": (
                        environment_contrastive_active / seen
                    ),
                    "train/environment_contrastive_queue_size": len(
                        environment_contrastive_queue.labels
                    ),
                    "train/environment_contrastive_weight": environment_contrastive_weight,
                }
            )
        if args.group_dro_eta > 0:
            entropy = -(group_dro_weights * group_dro_weights.clamp_min(1e-12).log()).sum()
            metrics.update(
                {
                    "train/group_dro_active_groups": int(group_dro_active.sum()),
                    "train/group_dro_weight_max": float(group_dro_weights.max()),
                    "train/group_dro_weight_min": float(group_dro_weights.min()),
                    "train/group_dro_weight_entropy": float(entropy),
                }
            )
        if args.hierarchical_loss_weight > 0:
            metrics["train/hierarchical_loss"] = hierarchical_loss_sum / seen
        if args.visual_channel_dropout > 0:
            metrics["train/visual_channel_dropout_fraction"] = (
                visual_channel_drops / max(visual_channel_opportunities, 1)
            )
        if args.structured_modality_dropout > 0:
            for branch, name in enumerate(branch_names):
                metrics[f"train/{name}_branch_dropout_fraction"] = (
                    int(branch_drops[branch]) / max(int(branch_opportunities[branch]), 1)
                )
        if args.temporal_scale_jitter > 0:
            metrics["train/temporal_scale_mean"] = float(batch["temporal_scale"].mean())
        if blend_teacher is not None:
            metrics["train/blend_loss"] = blend_loss_sum / seen
        if args.environment_auxiliary_weight > 0:
            metrics["train/environment_loss"] = environment_loss_sum / seen
        if args.masked_modality_distillation_weight > 0:
            metrics["train/masked_modality_distillation_loss"] = (
                masked_modality_distillation_loss_sum / seen
            )
        if args.mixup_alpha > 0:
            metrics["train/mixup_last_lambda"] = batch["mix_lambda"]
        if subject_head is not None:
            metrics.update({
                "train/subject_loss": subject_loss_sum / seen,
                "train/subject_accuracy": subject_correct / seen,
                "train/adversarial_coefficient": adversarial_coefficient,
            })
        if valid_loader is not None and (epoch % args.eval_every == 0 or epoch == args.epochs):
            validation, predictions = evaluate(ema, valid_loader, device, args)
            metrics.update({f"valid/{key}": value for key, value in validation.items()})
            state = {
                "model": {
                    name: value.detach().cpu().half() if value.is_floating_point() else value.detach().cpu()
                    for name, value in ema.state_dict().items()
                },
                "config": serializable,
                "epoch": epoch,
                "metrics": validation,
            }
            if args.save_epoch_checkpoints:
                epoch_directory = destination / "epochs"
                epoch_directory.mkdir(exist_ok=True)
                torch.save(state, epoch_directory / f"epoch_{epoch:02d}.pt")
                np.savez(epoch_directory / f"validation_{epoch:02d}.npz", **predictions)
            selection_accuracy = validation.get("selection_accuracy", validation["accuracy"])
            if selection_accuracy > best_accuracy:
                best_accuracy = selection_accuracy
                best_epoch = epoch
                torch.save(state, destination / "best.pt")
                np.savez(destination / "validation.npz", **predictions)
                (destination / "metrics.json").write_text(
                    json.dumps(validation | {"epoch": epoch}, indent=2) + "\n"
                )
        elif valid_loader is None and (args.save_epoch_checkpoints or epoch == args.epochs):
            state = {
                "model": {
                    name: value.detach().cpu().half() if value.is_floating_point() else value.detach().cpu()
                    for name, value in ema.state_dict().items()
                },
                "config": serializable,
                "epoch": epoch,
                "metrics": {},
            }
            if args.save_epoch_checkpoints:
                epoch_directory = destination / "epochs"
                epoch_directory.mkdir(exist_ok=True)
                torch.save(state, epoch_directory / f"epoch_{epoch:02d}.pt")
            if epoch == args.epochs:
                best_epoch = epoch
                torch.save(state, destination / "best.pt")
        metrics["valid/best_accuracy"] = best_accuracy
        with (destination / "history.jsonl").open("a") as handle:
            handle.write(json.dumps({"epoch": epoch} | metrics) + "\n")
        swanlab.log(metrics, step=epoch)
        print(
            f"epoch={epoch} " + " ".join(f"{name}={value:.5f}" for name, value in metrics.items()),
            flush=True,
        )
        if valid_loader is not None and epoch - best_epoch >= args.patience:
            print("Early stopping", flush=True)
            break

    checkpoint = torch.load(destination / "best.pt", map_location="cpu", weights_only=False)
    quantized, tensor_bytes = quantize_state_dict(checkpoint["model"])
    quantized_path = destination / "best_int8.pt"
    torch.save(
        {
            "model_int8": quantized,
            "config": serializable,
            "epoch": checkpoint["epoch"],
            "metrics": checkpoint["metrics"],
            "tensor_bytes": tensor_bytes,
        },
        quantized_path,
    )
    try:
        swanlab_url = run.url
    except ValueError:
        swanlab_url = None
    completion = {
        "status": "complete",
        "best_accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "checkpoint_bytes": (destination / "best.pt").stat().st_size,
        "quantized_checkpoint_bytes": quantized_path.stat().st_size,
        "quantized_tensor_bytes": tensor_bytes,
        "parameters": serializable["parameters"],
        "peak_cuda_mb": torch.cuda.max_memory_allocated() / 1e6,
        "elapsed_seconds": time.monotonic() - started,
        "swanlab_url": swanlab_url,
    }
    (destination / "completion.json").write_text(json.dumps(completion, indent=2) + "\n")
    swanlab.log(
        {
            "system/checkpoint_bytes": completion["checkpoint_bytes"],
            "system/quantized_checkpoint_bytes": completion["quantized_checkpoint_bytes"],
        }
    )
    swanlab.finish()
    if feature_hook is not None:
        feature_hook.remove()
    print("DONE " + json.dumps(completion), flush=True)


if __name__ == "__main__":
    main()

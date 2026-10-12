#!/usr/bin/env python3
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data import FOLDS, get_records


PAIRS = ((6, 16), (7, 17), (8, 18), (9, 19))
FEATURES = {
    "fused": ("fused",),
    "visual": ("depth", "thermal", "ir"),
    "all": ("fused", "depth", "thermal", "ir", "skeleton", "imu"),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--representation", choices=tuple(FEATURES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.08)
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize(values):
    return values / np.clip(np.linalg.norm(values, axis=1, keepdims=True), 1e-8, None)


def representation(archive, names):
    branches = []
    for name in names:
        center = archive[f"{name}_center"].astype(np.float32)
        flipped = archive[f"{name}_flip"].astype(np.float32)
        branches.append(normalize(center + flipped))
    return normalize(np.concatenate(branches, axis=1)).astype(np.float32)


class ResidualMetric(nn.Module):
    def __init__(self, dimension, rank):
        super().__init__()
        self.down = nn.Linear(dimension, rank, bias=False)
        self.up = nn.Linear(rank, dimension, bias=False)
        self.scale = nn.Parameter(torch.tensor(0.1))
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, values):
        residual = self.up(functional.gelu(self.down(values)))
        return functional.normalize(values + self.scale.tanh() * residual, dim=1)


def episode_loss(model, values, labels, users, source, target, temperature):
    source_rows = users == source
    target_rows = users == target
    source_values = model(values[source_rows])
    target_values = model(values[target_rows])
    source_labels = labels[source_rows]
    target_labels = labels[target_rows]
    classes = torch.unique(source_labels).sort().values
    prototypes = torch.stack(
        [source_values[source_labels == label].mean(dim=0) for label in classes]
    )
    prototypes = functional.normalize(prototypes, dim=1)
    available = torch.isin(target_labels, classes)
    if not available.any():
        return values.new_zeros(()), 0, 0
    targets = torch.searchsorted(classes, target_labels[available])
    logits = target_values[available] @ prototypes.T / temperature
    loss = functional.cross_entropy(logits, targets)
    correct = int((logits.argmax(dim=1) == targets).sum())
    return loss, int(available.sum()), correct


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    archive = np.load(args.features)
    records = get_records()
    indices = archive["indices"].astype(np.int64)
    labels_np = archive["labels"].astype(np.int64)
    users_np = np.asarray([records[int(index)]["user"] for index in indices])
    keep = ~np.isin(users_np, FOLDS[args.fold])
    values = torch.from_numpy(
        representation(archive, FEATURES[args.representation])[keep]
    ).cuda()
    labels = torch.from_numpy(labels_np[keep]).long().cuda()
    users = torch.from_numpy(users_np[keep]).long().cuda()
    training_pairs = [pair for pair in PAIRS if not set(pair) & set(FOLDS[args.fold])]
    model = ResidualMetric(values.shape[1], args.rank).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        noisy = functional.normalize(values + torch.randn_like(values) * args.noise, dim=1)
        losses = []
        rows = 0
        correct = 0
        for left, right in training_pairs:
            for source, target in ((left, right), (right, left)):
                loss, count, episode_correct = episode_loss(
                    model, noisy, labels, users, source, target, args.temperature
                )
                losses.append(loss)
                rows += count
                correct += episode_correct
        objective = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            values_epoch = {
                "epoch": epoch,
                "loss": float(objective.detach()),
                "training_accuracy": correct / rows,
                "scale": float(model.scale.tanh().detach()),
            }
            history.append(values_epoch)
            print(json.dumps(values_epoch), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            "config": vars(args) | {
                "features": str(args.features),
                "output": str(args.output),
                "dimension": values.shape[1],
                "training_pairs": training_pairs,
                "excluded_users": FOLDS[args.fold],
            },
            "history": history,
        },
        args.output,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import FOLDS, get_records
from src.r2plus1d_models import RadarPointTokens


class RadarDataset(Dataset):
    def __init__(self, indices, training):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.training = training
        self.records = get_records()
        self.points = np.load(
            ROOT / "data/processed/radar_points.npy", mmap_mode="r"
        )

    def __len__(self):
        return len(self.indices)

    def augment(self, values):
        valid = values[..., 9] > 0.5
        retained = valid & (torch.rand_like(values[..., 9]) > 0.08)
        values = values.clone()
        values[..., :9] += torch.randn_like(values[..., :9]) * 0.012
        spatial_scale = math.exp(random.uniform(-0.12, 0.12))
        values[..., :3] *= spatial_scale
        values[..., 6:9] *= spatial_scale
        values[..., 0] += random.uniform(-0.08, 0.08)
        values[..., 1] += random.uniform(-0.08, 0.08)
        values[..., 2] += random.uniform(-0.04, 0.04)
        values[..., 3] *= math.exp(random.uniform(-0.10, 0.10))
        values *= retained.unsqueeze(-1)
        values[..., 9] = retained
        if random.random() < 0.35:
            length = random.randint(3, 9)
            start = random.randint(0, values.shape[0] - length)
            values[start:start + length] = 0
        return values

    def __getitem__(self, position):
        index = int(self.indices[position])
        values = torch.from_numpy(np.asarray(self.points[index]).copy()).float()
        if self.training:
            values = self.augment(values)
        record = self.records[index]
        return {
            "radar": values,
            "label": int(record["label"]),
            "index": index,
            "user": int(record["user"]),
        }


class RadarPointExpert(nn.Module):
    def __init__(self, depth=2, dropout=0.2):
        super().__init__()
        dimension = 192
        self.encoder = RadarPointTokens(outputs=dimension)
        self.class_token = nn.Parameter(torch.zeros(1, 1, dimension))
        self.position_embedding = nn.Parameter(torch.zeros(1, 17, dimension))
        layer = nn.TransformerEncoderLayer(
            dimension,
            nhead=6,
            dim_feedforward=dimension * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=depth, norm=nn.LayerNorm(dimension)
        )
        self.head = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension, 40),
        )
        self.temporal_head = nn.Linear(dimension, 40)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, radar):
        tokens = self.encoder(radar)
        class_token = self.class_token.expand(len(radar), -1, -1)
        fused = self.transformer(
            torch.cat([class_token, tokens], dim=1) + self.position_embedding
        )
        return self.head(fused[:, 0]), self.temporal_head(tokens.mean(dim=1))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--auxiliary-weight", type=float, default=0.2)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--swanlab-mode", choices=("cloud", "local", "disabled"), default="cloud")
    return parser.parse_args()


def soft_cross_entropy(logits, targets, label_smoothing):
    targets = targets * (1 - label_smoothing) + label_smoothing / targets.shape[1]
    return -(targets * functional.log_softmax(logits, dim=1)).sum(dim=1).mean()


def mixup(radar, labels, alpha):
    targets = functional.one_hot(labels, 40).float()
    if alpha <= 0:
        return radar, targets
    coefficient = float(np.random.beta(alpha, alpha))
    permutation = torch.randperm(len(radar), device=radar.device)
    mixed = coefficient * radar + (1 - coefficient) * radar[permutation]
    mixed_targets = coefficient * targets + (1 - coefficient) * targets[permutation]
    return mixed, mixed_targets


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    probabilities = []
    labels = []
    indices = []
    users = []
    losses = []
    for batch in loader:
        radar = batch["radar"].to(device, non_blocking=True)
        target = batch["label"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, auxiliary = model(radar)
            loss = functional.cross_entropy(logits, target)
            loss += 0.2 * functional.cross_entropy(auxiliary, target)
        probabilities.append(logits.float().softmax(dim=1).cpu().numpy())
        labels.append(batch["label"].numpy())
        indices.append(batch["index"].numpy())
        users.append(batch["user"].numpy())
        losses.append(float(loss) * len(target))
    probabilities = np.concatenate(probabilities)
    labels = np.concatenate(labels)
    indices = np.concatenate(indices)
    users = np.concatenate(users)
    prediction = probabilities.argmax(axis=1)
    user_accuracy = {
        str(user): float((prediction[users == user] == labels[users == user]).mean())
        for user in sorted(set(users.tolist()))
    }
    return {
        "loss": sum(losses) / len(labels),
        "accuracy": float((prediction == labels).mean()),
        "correct": int((prediction == labels).sum()),
        "rows": len(labels),
        "user_macro_accuracy": float(np.mean(list(user_accuracy.values()))),
        "user_accuracy": user_accuracy,
    }, probabilities, labels, indices, users


def main():
    args = parse_args()
    random.seed(args.seed + args.fold)
    np.random.seed(args.seed + args.fold)
    torch.manual_seed(args.seed + args.fold)
    torch.cuda.manual_seed_all(args.seed + args.fold)
    device = torch.device("cuda")
    records = get_records()
    available = np.load(ROOT / "data/processed/radar_points_mask.npy").astype(bool)
    train_indices = [
        record["index"]
        for record in records
        if record["split"] == "train"
        and record["user"] not in FOLDS[args.fold]
        and available[record["index"]]
    ]
    valid_indices = [
        record["index"]
        for record in records
        if record["split"] == "train"
        and record["user"] in FOLDS[args.fold]
        and available[record["index"]]
    ]
    if set(records[index]["user"] for index in train_indices) & set(
        records[index]["user"] for index in valid_indices
    ):
        raise RuntimeError("Subject leakage")
    if set(records[index]["label"] for index in train_indices) != set(range(40)):
        raise RuntimeError("Radar training split is missing classes")

    generator = torch.Generator().manual_seed(args.seed + args.fold)
    train_loader = DataLoader(
        RadarDataset(train_indices, training=True),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        generator=generator,
    )
    valid_loader = DataLoader(
        RadarDataset(valid_indices, training=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    model = RadarPointExpert().to(device)
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps = args.epochs * len(train_loader)
    warmup_steps = len(train_loader)
    global_step = 0
    destination = HERE / "outputs" / args.name
    destination.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {
        "train_rows": len(train_indices),
        "valid_rows": len(valid_indices),
        "train_users": sorted({records[index]["user"] for index in train_indices}),
        "valid_users": sorted({records[index]["user"] for index in valid_indices}),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "architecture": "PointNet-style 1x1 CNN + dilated temporal CNN + Transformer",
        "precision": "BF16 autocast; FP32 optimizer and EMA",
    }
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    import swanlab

    run = swanlab.init(
        project="cuhkx-small-multimodal",
        name=args.name,
        config=config,
        public=False,
        mode=args.swanlab_mode,
        log_dir=str(HERE / "swanlog"),
        description="Strict cross-user radar point-cloud expert ablation.",
    )
    best_correct = -1
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for batch in train_loader:
            radar = batch["radar"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            radar, targets = mixup(radar, labels, args.mixup_alpha)
            if global_step < warmup_steps:
                rate = args.lr * (global_step + 1) / warmup_steps
            else:
                progress = (global_step - warmup_steps) / max(1, total_steps - warmup_steps)
                rate = args.lr * 0.5 * (1 + math.cos(math.pi * progress))
            optimizer.param_groups[0]["lr"] = rate
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, auxiliary = model(radar)
                loss = soft_cross_entropy(logits, targets, args.label_smoothing)
                loss += args.auxiliary_weight * soft_cross_entropy(
                    auxiliary, targets, args.label_smoothing
                )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            with torch.no_grad():
                for target, source in zip(ema.parameters(), model.parameters()):
                    target.lerp_(source, 1 - args.ema_decay)
                for target, source in zip(ema.buffers(), model.buffers()):
                    target.copy_(source)
            loss_sum += float(loss.detach()) * len(labels)
            seen += len(labels)
            global_step += 1
        metrics, probabilities, labels, indices, users = evaluate(
            ema, valid_loader, device
        )
        values = {
            "epoch": epoch,
            "train/loss": loss_sum / seen,
            "train/lr": optimizer.param_groups[0]["lr"],
            **{f"valid/{key}": value for key, value in metrics.items() if key != "user_accuracy"},
        }
        swanlab.log(values, step=epoch)
        print(" ".join(f"{key}={value:.5f}" if isinstance(value, float) else f"{key}={value}" for key, value in values.items()), flush=True)
        if metrics["correct"] > best_correct:
            best_correct = metrics["correct"]
            torch.save(
                {"model": ema.state_dict(), "config": config, "epoch": epoch, "metrics": metrics},
                destination / "best.pt",
            )
            np.savez(
                destination / "validation.npz",
                probabilities=probabilities,
                labels=labels,
                indices=indices,
                users=users,
            )
            (destination / "metrics.json").write_text(
                json.dumps(metrics | {"epoch": epoch}, indent=2) + "\n"
            )
    completion = {
        "status": "complete",
        "best_correct": best_correct,
        "valid_rows": len(valid_indices),
        "elapsed_seconds": time.monotonic() - started,
        "checkpoint_bytes": (destination / "best.pt").stat().st_size,
        "swanlab_url": run.url if hasattr(run, "url") else None,
    }
    (destination / "completion.json").write_text(json.dumps(completion, indent=2) + "\n")
    print("DONE " + json.dumps(completion), flush=True)
    swanlab.finish()


if __name__ == "__main__":
    main()

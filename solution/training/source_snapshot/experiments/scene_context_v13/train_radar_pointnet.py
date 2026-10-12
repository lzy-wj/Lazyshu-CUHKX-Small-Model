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
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data import get_indices, get_records


class RadarDataset(Dataset):
    def __init__(self, indices, training=False):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.training = training
        self.records = get_records()
        self.points = np.load(ROOT / "data/processed/radar_points.npy", mmap_mode="r")
        self.available = np.load(ROOT / "data/processed/radar_points_mask.npy", mmap_mode="r")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = int(self.indices[position])
        points = torch.from_numpy(np.asarray(self.points[index], dtype=np.float32).copy())
        if self.training:
            valid = points[..., 9] > 0
            keep = (torch.rand(valid.shape) > 0.1) & valid
            empty = keep.sum(dim=1) == 0
            keep[empty] = valid[empty]
            points[..., :9] += torch.randn_like(points[..., :9]) * 0.01 * valid.unsqueeze(-1)
            points[..., 9] = keep
            start = random.randint(0, 7)
            end = random.randint(56, 64)
            points = functional.interpolate(
                points[start:end].permute(2, 1, 0).unsqueeze(0),
                size=(points.shape[1], 64),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).permute(2, 1, 0)
            points[..., 9] = (points[..., 9] > 0.5).float()
        return {
            "points": points,
            "available": torch.tensor(self.available[index], dtype=torch.float32),
            "label": self.records[index]["label"],
            "index": index,
        }


class RadarPointNet(nn.Module):
    def __init__(self, width=192, dropout=0.25):
        super().__init__()
        self.point_encoder = nn.Sequential(
            nn.Linear(9, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        self.frame_projection = nn.Sequential(
            nn.Linear(256, width), nn.LayerNorm(width), nn.GELU()
        )
        self.temporal_convolution = nn.Sequential(
            nn.Conv1d(width, width, 5, padding=2, groups=width, bias=False),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Conv1d(width, width, 3, padding=2, dilation=2, groups=width, bias=False),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Conv1d(width, width, 1, bias=False),
        )
        layer = nn.TransformerEncoderLayer(
            width,
            nhead=6,
            dim_feedforward=width * 3,
            dropout=0.15,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            layer, num_layers=2, enable_nested_tensor=False
        )
        self.class_token = nn.Parameter(torch.zeros(1, 1, width))
        self.position_embedding = nn.Parameter(torch.zeros(1, 65, width))
        self.normalization = nn.LayerNorm(width)
        self.head = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, 40)
        )
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, points):
        point_mask = points[..., 9].bool()
        encoded = self.point_encoder(points[..., :9])
        maximum = encoded.masked_fill(~point_mask.unsqueeze(-1), -1e4).amax(dim=2)
        maximum = torch.where(point_mask.any(dim=2, keepdim=True), maximum, torch.zeros_like(maximum))
        mean = (encoded * point_mask.unsqueeze(-1)).sum(dim=2) / point_mask.sum(
            dim=2, keepdim=True
        ).clamp_min(1)
        frames = self.frame_projection(torch.cat([mean, maximum], dim=-1))
        frames = frames + self.temporal_convolution(frames.transpose(1, 2)).transpose(1, 2)
        token = self.class_token.expand(len(points), -1, -1)
        values = torch.cat([token, frames], dim=1) + self.position_embedding
        values = self.temporal_transformer(values)
        return self.head(self.normalization(values[:, 0]))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--fold", type=int, choices=(-1, 0, 1), required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--swanlab-mode", choices=("online", "offline", "disabled"), default="online")
    return parser.parse_args()


def evaluate(model, loader, device, prior):
    model.eval()
    probabilities, labels, indices, available = [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            points = batch["points"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(points)
            current = logits.float().softmax(dim=1).cpu().numpy()
            mask = batch["available"].numpy().astype(bool)
            current[~mask] = prior
            probabilities.append(current)
            labels.append(batch["label"].numpy())
            indices.append(batch["index"].numpy())
            available.append(mask)
    probabilities = np.concatenate(probabilities)
    labels = np.concatenate(labels)
    indices = np.concatenate(indices)
    available = np.concatenate(available)
    prediction = probabilities.argmax(axis=1)
    return {
        "accuracy": float((prediction == labels).mean()),
        "available_accuracy": float((prediction[available] == labels[available]).mean()),
        "coverage": float(available.mean()),
        "nll": float(-np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-8, 1)).mean()),
    }, {
        "probabilities": probabilities,
        "labels": labels,
        "indices": indices,
        "available": available,
    }


def learning_rate(epoch, epochs, maximum):
    warmup = min(epoch / 5, 1)
    cosine = 0.05 + 0.95 * (1 + math.cos(math.pi * epoch / epochs)) / 2
    return maximum * warmup * cosine


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    records = get_records()
    train_indices, valid_indices = get_indices(records, args.fold)
    available = np.load(ROOT / "data/processed/radar_points_mask.npy") > 0
    train_indices = [index for index in train_indices if available[index]]
    labels = np.asarray([records[index]["label"] for index in train_indices])
    counts = np.bincount(labels, minlength=40).clip(1)
    prior = (counts + 1) / (counts.sum() + 40)
    weights = (counts.mean() / counts) ** 0.35
    weights = torch.tensor(weights / weights.mean(), device=device, dtype=torch.float32)
    options = {
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(
        RadarDataset(train_indices, training=True),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed),
        **options,
    )
    valid_loader = (
        DataLoader(RadarDataset(valid_indices), batch_size=args.batch_size, shuffle=False, **options)
        if valid_indices
        else None
    )
    model = RadarPointNet().to(device)
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    destination = ROOT / "experiments/scene_context_v13/outputs/checkpoints" / args.name
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "best.pt").is_file():
        raise RuntimeError("Experiment already completed")
    config = vars(args) | {
        "architecture": "PointNet frame encoder + temporal CNN + Transformer",
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train_clips": len(train_indices),
        "valid_clips": len(valid_indices),
        "visible_gpus": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "radar_manifest_sha256": hashlib.sha256(
            (ROOT / "data/processed/radar_points_manifest.json").read_bytes()
        ).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    import swanlab

    run = swanlab.init(
        project="cuhkx-small-multimodal",
        name=args.name,
        config=config,
        public=False,
        mode=args.swanlab_mode,
        log_dir=str(ROOT / "outputs/swanlog"),
        description="Raw mmWave PointNet plus temporal CNN/Transformer; subject-disjoint validation.",
    )
    best_accuracy = -1.0
    best_epoch = 0
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        rate = learning_rate(epoch, args.epochs, args.lr)
        for group in optimizer.param_groups:
            group["lr"] = rate
        loss_sum = 0.0
        correct = 0
        seen = 0
        for batch in train_loader:
            points = batch["points"].to(device, non_blocking=True)
            target = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(points)
                loss = functional.cross_entropy(logits, target, weight=weights, label_smoothing=0.05)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            with torch.no_grad():
                decay = min(0.99, 1 - 1 / (epoch * len(train_loader) + seen // args.batch_size + 1))
                for target_parameter, source_parameter in zip(ema.parameters(), model.parameters()):
                    target_parameter.lerp_(source_parameter, 1 - decay)
                for target_buffer, source_buffer in zip(ema.buffers(), model.buffers()):
                    target_buffer.copy_(source_buffer)
            loss_sum += loss.item() * len(target)
            correct += int((logits.argmax(dim=1) == target).sum())
            seen += len(target)
        metrics = {
            "train/loss": loss_sum / seen,
            "train/accuracy": correct / seen,
            "train/lr": rate,
        }
        if valid_loader is not None:
            validation, predictions = evaluate(ema, valid_loader, device, prior)
            metrics.update({f"valid/{key}": value for key, value in validation.items()})
            if validation["available_accuracy"] > best_accuracy:
                best_accuracy = validation["available_accuracy"]
                best_epoch = epoch
                torch.save(
                    {
                        "model": {name: value.detach().cpu() for name, value in ema.state_dict().items()},
                        "config": config,
                        "epoch": epoch,
                        "metrics": validation,
                        "prior": prior,
                    },
                    destination / "best.pt",
                )
                np.savez(destination / "validation.npz", **predictions)
                (destination / "metrics.json").write_text(json.dumps(validation | {"epoch": epoch}, indent=2) + "\n")
        elif epoch == args.epochs:
            torch.save(
                {
                    "model": {name: value.detach().cpu() for name, value in ema.state_dict().items()},
                    "config": config,
                    "epoch": epoch,
                    "metrics": {},
                    "prior": prior,
                },
                destination / "best.pt",
            )
        with (destination / "history.jsonl").open("a") as handle:
            handle.write(json.dumps({"epoch": epoch} | metrics) + "\n")
        swanlab.log(metrics, step=epoch)
        print(f"epoch={epoch} " + " ".join(f"{key}={value:.5f}" for key, value in metrics.items()), flush=True)
        if valid_loader is not None and epoch - best_epoch >= args.patience:
            break
    completion = {
        "status": "complete",
        "best_available_accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "elapsed_seconds": time.monotonic() - started,
    }
    (destination / "completion.json").write_text(json.dumps(completion, indent=2) + "\n")
    swanlab.finish()


if __name__ == "__main__":
    main()

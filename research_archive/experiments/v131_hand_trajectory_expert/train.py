#!/usr/bin/env python3
import argparse
import copy
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.v131_hand_trajectory_expert.model import HandTrajectoryExpert
from src.data import get_indices, get_records


class SkeletonDataset(Dataset):
    def __init__(self, indices):
        self.indices = indices
        self.values = np.load(ROOT / "data/processed/skeleton.npy", mmap_mode="r")
        self.records = get_records()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = self.indices[position]
        return {
            "values": np.array(self.values[index]),
            "label": self.records[index]["label"],
            "index": index,
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--swanlab-mode", choices=("cloud", "local", "disabled"), default="cloud")
    return parser.parse_args()


def augment(values):
    batch = values.shape[0]
    pose = values[:, :, :51].reshape(batch, 64, 17, 3).clone()
    angle = torch.randn(batch, 1, 1, device=values.device) * 0.18
    horizontal = pose[..., 0] * angle.cos() - pose[..., 1] * angle.sin()
    vertical = pose[..., 0] * angle.sin() + pose[..., 1] * angle.cos()
    pose[..., 0], pose[..., 1] = horizontal, vertical
    pose = pose * (1 + torch.randn(batch, 1, 1, 1, device=values.device) * 0.06)
    pose = pose + torch.randn_like(pose) * 0.004
    output = values.clone()
    output[:, :, :51] = pose.flatten(2)
    return output


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    logits, labels, indices = [], [], []
    for batch in loader:
        values = batch["values"].cuda(non_blocking=True).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(values)
        logits.append(output.float().cpu())
        labels.append(batch["label"])
        indices.append(batch["index"])
    logits = torch.cat(logits)
    labels = torch.cat(labels).numpy()
    indices = torch.cat(indices).numpy()
    probabilities = logits.softmax(1).numpy()
    return {
        "accuracy": float((probabilities.argmax(1) == labels).mean()),
        "nll": float(-np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1)).mean()),
    }, {"probabilities": probabilities, "labels": labels, "indices": indices}


def main():
    args = parse_args()
    random.seed(args.seed + args.fold)
    np.random.seed(args.seed + args.fold)
    torch.manual_seed(args.seed + args.fold)
    torch.cuda.manual_seed_all(args.seed + args.fold)
    torch.backends.cuda.matmul.allow_tf32 = True
    records = get_records()
    train_indices, valid_indices = get_indices(records, args.fold)
    mask = np.load(ROOT / "data/processed/sensor_mask.npy")[:, 0] > 0
    train_indices = [index for index in train_indices if mask[index]]
    valid_indices = [index for index in valid_indices if mask[index]]
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(SkeletonDataset(train_indices), shuffle=True, **loader_args)
    valid_loader = DataLoader(SkeletonDataset(valid_indices), shuffle=False, **loader_args)
    model = HandTrajectoryExpert().cuda()
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0012, weight_decay=0.01)
    counts = np.bincount([records[index]["label"] for index in train_indices], minlength=40).clip(1)
    class_weights = torch.tensor(
        (counts.mean() / counts) ** 0.25, dtype=torch.float32, device="cuda"
    )
    destination = HERE / "outputs" / f"v131_hand_f{args.fold}"
    destination.mkdir(parents=True, exist_ok=False)
    config = {
        **vars(args),
        "train_rows": len(train_indices),
        "valid_rows": len(valid_indices),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "architecture": "body-normalized hand trajectory CNN + bidirectional GRU + Transformer",
        "model_sha256": hashlib.sha256((HERE / "model.py").read_bytes()).hexdigest(),
    }
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    import swanlab
    run = swanlab.init(
        project="cuhkx-small-multimodal",
        name=f"v131_hand_f{args.fold}",
        config=config,
        public=False,
        mode=args.swanlab_mode,
        log_dir=str(HERE / "swanlog"),
        description="CNN/GRU/Transformer hand-trajectory expert with subject-disjoint validation.",
    )
    best_accuracy = -1.0
    best_epoch = 0
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        learning_rate = 0.0012 * (0.05 + 0.95 * (1 + math.cos(math.pi * epoch / args.epochs)) / 2)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        loss_total = 0.0
        correct = 0
        seen = 0
        for batch in train_loader:
            values = augment(batch["values"].cuda(non_blocking=True).float())
            labels = batch["label"].cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(values)
                loss = functional.cross_entropy(
                    logits, labels, weight=class_weights, label_smoothing=0.05
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            with torch.no_grad():
                decay = min(0.99, 1 - 1 / (epoch * len(train_loader)))
                for target, source in zip(ema.parameters(), model.parameters()):
                    target.lerp_(source, 1 - decay)
            loss_total += float(loss.detach()) * len(labels)
            correct += int((logits.argmax(1) == labels).sum())
            seen += len(labels)
        metrics, predictions = evaluate(ema, valid_loader)
        values = {
            "epoch": epoch,
            "train/loss": loss_total / seen,
            "train/accuracy": correct / seen,
            "train/lr": learning_rate,
            "valid/accuracy": metrics["accuracy"],
            "valid/nll": metrics["nll"],
        }
        print(json.dumps(values), flush=True)
        swanlab.log({key: value for key, value in values.items() if key != "epoch"}, step=epoch)
        with (destination / "history.jsonl").open("a") as handle:
            handle.write(json.dumps(values) + "\n")
        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            best_epoch = epoch
            torch.save(
                {
                    "model": {name: value.detach().half().cpu() for name, value in ema.state_dict().items()},
                    "config": config,
                    "epoch": epoch,
                    "metrics": metrics,
                },
                destination / "best.pt",
            )
            np.savez(destination / "validation.npz", **predictions)
            (destination / "metrics.json").write_text(
                json.dumps(metrics | {"epoch": epoch}, indent=2) + "\n"
            )
        if epoch - best_epoch >= 8:
            print("Early stopping", flush=True)
            break
    swanlab.finish()
    print(
        "DONE "
        + json.dumps(
            {
                "best_accuracy": best_accuracy,
                "best_epoch": best_epoch,
                "elapsed_seconds": time.monotonic() - started,
                "checkpoint_bytes": (destination / "best.pt").stat().st_size,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()


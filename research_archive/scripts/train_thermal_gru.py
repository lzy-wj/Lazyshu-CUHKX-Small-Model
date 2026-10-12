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
from torch.nn import functional
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import FOLDS, ROOT, get_indices, get_records
from src.r2plus1d_models import quantize_state_dict
from src.thermal_gru import ThermalGRUNet


class ThermalDataset(Dataset):
    def __init__(self, indices, training=False, frames=16, image_size=112):
        self.indices = indices
        self.training = training
        self.frames = frames
        self.image_size = image_size
        self.records = get_records()
        self.mask = np.load(ROOT / "data/processed/video_mask.npy", mmap_mode="r")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = self.indices[position]
        video = np.load(ROOT / f"data/processed/video_24_160/{index:05d}.npy", mmap_mode="r")
        edges = np.linspace(0, len(video), self.frames + 1)
        selected = []
        for left, right in zip(edges[:-1], edges[1:]):
            low = min(int(left), len(video) - 1)
            high = min(max(int(np.ceil(right)) - 1, low), len(video) - 1)
            selected.append(random.randint(low, high) if self.training else (low + high) // 2)
        values = torch.from_numpy(np.asarray(video[selected, 4:7]).copy()).float().div_(255)
        if self.training and random.random() < 0.5:
            values = torch.flip(values, dims=(-1,))
        values = functional.interpolate(
            values, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
        )
        return {
            "thermal": values.sub_(0.5).div_(0.25),
            "label": int(self.records[index]["label"]),
            "index": index,
            "available": float(self.mask[index, 2]),
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--swanlab-mode", choices=("cloud", "local", "disabled"), default="cloud")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    probabilities, labels, indices, availability = [], [], [], []
    for batch in loader:
        video = batch["thermal"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(video)
        probabilities.append(logits.float().softmax(1).cpu())
        labels.append(batch["label"])
        indices.append(batch["index"])
        availability.append(batch["available"])
    probabilities = torch.cat(probabilities).numpy()
    labels = torch.cat(labels).numpy()
    indices = torch.cat(indices).numpy()
    availability = torch.cat(availability).numpy().astype(bool)
    return {
        "accuracy": float((probabilities.argmax(1) == labels).mean()),
        "available_accuracy": float((probabilities[availability].argmax(1) == labels[availability]).mean()),
        "coverage": float(availability.mean()),
    }, {"probabilities": probabilities, "labels": labels, "indices": indices}


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Thermal training requires CUDA")
    seed_everything(args.seed + args.fold)
    records = get_records()
    train_indices, valid_indices = get_indices(records, args.fold)
    common = {
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(
        ThermalDataset(train_indices, training=True),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed + args.fold),
        **common,
    )
    valid_loader = DataLoader(
        ThermalDataset(valid_indices), batch_size=args.batch_size, shuffle=False, **common
    )
    device = torch.device("cuda")
    model = ThermalGRUNet().to(device)
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    destination = ROOT / "outputs/checkpoints" / args.name
    destination.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train_clips": len(train_indices),
        "valid_clips": len(valid_indices),
        "validation_users": FOLDS[args.fold],
        "initialization": "random",
        "architecture": "2D residual CNN + 2-layer bidirectional GRU + temporal attention",
        "input": "16 Thermal frames sampled from the 24-frame full-view 160px cache, resized to 112px",
        "precision": "BF16 autocast",
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ("scripts/train_thermal_gru.py", "src/thermal_gru.py", "data/metadata/samples.json")
        },
    }
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (destination / "split.json").write_text(json.dumps({"train": train_indices, "valid": valid_indices}) + "\n")
    import swanlab
    run = swanlab.init(
        project="cuhkx-small-multimodal",
        name=args.name,
        config=config,
        public=False,
        mode=args.swanlab_mode,
        log_dir=str(ROOT / "outputs/swanlog"),
        description="Scratch Thermal CNN-BiGRU expert; user-isolated validation.",
    )
    best_accuracy = -1.0
    best_epoch = 0
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_started = time.monotonic()
        rate = args.lr * 0.5 * (1 + math.cos(math.pi * (epoch - 1) / args.epochs))
        for group in optimizer.param_groups:
            group["lr"] = rate
        loss_sum = 0
        correct = 0
        seen = 0
        for batch in train_loader:
            video = batch["thermal"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(video)
                loss = functional.cross_entropy(logits, labels, label_smoothing=0.1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            with torch.no_grad():
                decay = min(0.99, 1 - 1 / (seen / args.batch_size + 1))
                for target, source in zip(ema.parameters(), model.parameters()):
                    target.lerp_(source, 1 - decay)
                for target, source in zip(ema.buffers(), model.buffers()):
                    target.copy_(source)
            count = len(labels)
            loss_sum += float(loss.detach()) * count
            correct += int((logits.argmax(1) == labels).sum())
            seen += count
        validation, predictions = evaluate(ema, valid_loader, device)
        metrics = {
            "train/loss": loss_sum / seen,
            "train/accuracy": correct / seen,
            "train/lr": rate,
            "valid/accuracy": validation["accuracy"],
            "valid/available_accuracy": validation["available_accuracy"],
            "valid/coverage": validation["coverage"],
            "system/epoch_seconds": time.monotonic() - epoch_started,
            "system/peak_cuda_mb": torch.cuda.max_memory_allocated() / 1e6,
        }
        if validation["accuracy"] > best_accuracy:
            best_accuracy = validation["accuracy"]
            best_epoch = epoch
            torch.save(
                {
                    "model": {
                        name: value.detach().cpu().half() if value.is_floating_point() else value.detach().cpu()
                        for name, value in ema.state_dict().items()
                    },
                    "config": config,
                    "epoch": epoch,
                    "metrics": validation,
                },
                destination / "best.pt",
            )
            np.savez(destination / "validation.npz", **predictions)
            (destination / "metrics.json").write_text(json.dumps(validation | {"epoch": epoch}, indent=2) + "\n")
        metrics["valid/best_accuracy"] = best_accuracy
        with (destination / "history.jsonl").open("a") as handle:
            handle.write(json.dumps({"epoch": epoch} | metrics) + "\n")
        swanlab.log(metrics, step=epoch)
        print(f"epoch={epoch} " + " ".join(f"{key}={value:.5f}" for key, value in metrics.items()), flush=True)
        if epoch - best_epoch >= args.patience:
            print("Early stopping", flush=True)
            break
    checkpoint = torch.load(destination / "best.pt", map_location="cpu", weights_only=False)
    quantized, tensor_bytes = quantize_state_dict(checkpoint["model"])
    torch.save(
        {"model_int8": quantized, "config": config, "epoch": checkpoint["epoch"], "metrics": checkpoint["metrics"]},
        destination / "best_int8.pt",
    )
    completion = {
        "best_accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "elapsed_seconds": time.monotonic() - started,
        "checkpoint_bytes": (destination / "best.pt").stat().st_size,
        "quantized_checkpoint_bytes": (destination / "best_int8.pt").stat().st_size,
        "quantized_tensor_bytes": tensor_bytes,
    }
    (destination / "completion.json").write_text(json.dumps(completion, indent=2) + "\n")
    swanlab.finish()
    print("DONE " + json.dumps(completion), flush=True)


if __name__ == "__main__":
    main()

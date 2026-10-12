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
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import get_indices, get_records


class IMUDataset(Dataset):
    def __init__(self, indices, values, records):
        self.indices = indices
        self.values = values
        self.records = records

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = self.indices[position]
        return {
            "values": np.asarray(self.values[index]),
            "label": self.records[index]["label"],
            "index": index,
        }


class PositionalEncoding(nn.Module):
    def __init__(self, dimension=128, length=50):
        super().__init__()
        positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(torch.arange(0, dimension, 2).float() * (-math.log(10000) / dimension))
        encoding = torch.zeros(length, dimension)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies)
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, values):
        return values + self.encoding[:, :values.shape[1]]


class OfficialIMUTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.convolution = nn.Sequential(
            nn.Conv1d(45, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.position = PositionalEncoding()
        layer = nn.TransformerEncoderLayer(128, 2, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, 2)
        self.classifier = nn.Linear(128 * 50, 40)

    def forward(self, values):
        values = self.convolution(values.transpose(1, 2)).transpose(1, 2)
        values = self.transformer(self.position(values))
        return self.classifier(values.flatten(1))


def evaluate(model, loader, device, available):
    model.eval()
    probabilities, labels, indices = [], [], []
    with torch.inference_mode():
        for batch in loader:
            values = batch["values"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(values)
            probabilities.append(logits.float().softmax(1).cpu().numpy())
            labels.append(batch["label"].numpy())
            indices.append(batch["index"].numpy())
    probabilities = np.concatenate(probabilities)
    labels = np.concatenate(labels)
    indices = np.concatenate(indices)
    prediction = probabilities.argmax(1)
    present = available[indices]
    return {
        "accuracy": float((prediction == labels).mean()),
        "available_accuracy": float((prediction[present] == labels[present]).mean()),
        "available_rows": int(present.sum()),
        "nll": float(-np.log(probabilities[np.arange(len(labels)), labels].clip(1e-8)).mean()),
    }, {"probabilities": probabilities, "labels": labels, "indices": indices}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=(0, 1))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu", type=int, required=True, choices=(4, 5, 6, 7))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")

    records = get_records()
    train_indices, valid_indices = get_indices(records, args.fold)
    sensor_mask = np.load(ROOT / "data/processed/sensor_mask.npy")[:, 1] > 0
    train_indices = [index for index in train_indices if sensor_mask[index]]
    values = np.load(ROOT / "data/processed/imu_official_exact.npy", mmap_mode="r")
    loader_options = dict(num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    train_loader = DataLoader(
        IMUDataset(train_indices, values, records),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(args.seed),
        **loader_options,
    )
    valid_loader = DataLoader(
        IMUDataset(valid_indices, values, records),
        batch_size=args.batch_size,
        shuffle=False,
        **loader_options,
    )
    model = OfficialIMUTransformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)
    criterion = nn.CrossEntropyLoss()

    name = f"v195_official_exact_s{args.seed}_f{args.fold}"
    destination = HERE / "outputs/checkpoints" / name
    destination.mkdir(parents=True, exist_ok=False)
    config = vars(args) | {
        "name": name,
        "train_clips": len(train_indices),
        "valid_clips": len(valid_indices),
        "validation_users": sorted({records[index]["user"] for index in valid_indices}),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "precision": "BF16 autocast; FP32 optimizer",
        "architecture": "official three-layer 1D CNN plus two-layer Transformer",
        "initialization": "random",
        "cache_sha256": hashlib.sha256((ROOT / "data/processed/imu_official_exact.npy").read_bytes()).hexdigest(),
    }
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (destination / "split.json").write_text(json.dumps({"train": train_indices, "valid": valid_indices}) + "\n")

    import swanlab

    run = swanlab.init(
        project="cuhkx-small-multimodal",
        name=name,
        config=config,
        public=False,
        mode="cloud",
        log_dir=str(HERE / "outputs/swanlog"),
        description="Exact official random-init IMU CNN-Transformer on subject-held-out validation",
    )
    best_accuracy = -1.0
    best_epoch = 0
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        correct = 0
        seen = 0
        loss_sum = 0.0
        epoch_started = time.monotonic()
        for batch in train_loader:
            values_batch = batch["values"].to(device, non_blocking=True)
            labels_batch = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(values_batch)
                loss = criterion(logits, labels_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            count = len(labels_batch)
            loss_sum += float(loss) * count
            correct += int((logits.argmax(1) == labels_batch).sum())
            seen += count
        metrics, predictions = evaluate(model, valid_loader, device, sensor_mask)
        scheduler.step(metrics["nll"])
        log = {
            "epoch": epoch,
            "train_loss": loss_sum / seen,
            "train_accuracy": correct / seen,
            "valid_accuracy": metrics["accuracy"],
            "valid_available_accuracy": metrics["available_accuracy"],
            "valid_nll": metrics["nll"],
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.monotonic() - epoch_started,
        }
        with (destination / "history.jsonl").open("a") as handle:
            handle.write(json.dumps(log) + "\n")
        swanlab.log(log, step=epoch)
        print(" ".join(f"{key}={value:.5f}" if isinstance(value, float) else f"{key}={value}" for key, value in log.items()), flush=True)
        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            best_epoch = epoch
            torch.save({"model": copy.deepcopy(model.state_dict()), "config": config, "epoch": epoch, "metrics": metrics}, destination / "best.pt")
            np.savez(destination / "validation.npz", **predictions)
            (destination / "metrics.json").write_text(json.dumps(metrics | {"epoch": epoch}, indent=2) + "\n")
    completion = {
        "status": "complete",
        "best_accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "checkpoint_bytes": (destination / "best.pt").stat().st_size,
        "elapsed_seconds": time.monotonic() - started,
    }
    (destination / "completion.json").write_text(json.dumps(completion, indent=2) + "\n")
    print(f"DONE {json.dumps(completion)}", flush=True)
    swanlab.finish()


if __name__ == "__main__":
    main()

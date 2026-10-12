#!/usr/bin/env python3
import argparse
import copy
import json
import math
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

from experiments.v110_object_token_transformer.model import ObjectTokenTransformer
from src.data import FOLDS, get_indices, get_records


class DetectionDataset(Dataset):
    def __init__(self, indices, feature_path, records, training=False):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.features = np.load(feature_path, mmap_mode="r")
        self.records = records
        self.training = training

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = int(self.indices[position])
        features = torch.from_numpy(np.asarray(self.features[index]).astype(np.float32))
        if self.training:
            present = features[..., 0] > 0
            drop = (torch.rand_like(features[..., 0]) < 0.12) & present
            features[drop] = 0
            features[..., 0] *= 1 + torch.randn_like(features[..., 0]) * 0.05
            features[..., 1:5] += (
                torch.randn_like(features[..., 1:5]) * 0.01 * present.unsqueeze(-1)
            )
            features.clamp_(0, 1)
        return {
            "features": features,
            "label": self.records[index]["label"],
            "index": index,
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--swanlab-mode", choices=("cloud", "local", "disabled"), default="cloud")
    parser.add_argument(
        "--features",
        type=Path,
        default=ROOT / "data/processed/yolo_object_tokens_8x3/features.float16.npy",
    )
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    probabilities = []
    labels = []
    indices = []
    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(features)
        probabilities.append(logits.float().softmax(1).cpu())
        labels.append(batch["label"])
        indices.append(batch["index"])
    return {
        "probabilities": torch.cat(probabilities).numpy(),
        "labels": torch.cat(labels).numpy(),
        "indices": torch.cat(indices).numpy(),
    }


def main():
    args = parse_args()
    physical_device = int(args.device)
    if physical_device not in range(4, 8):
        raise ValueError("V110 may only use GPUs 4-7")
    torch.manual_seed(args.seed + args.fold)
    np.random.seed(args.seed + args.fold)
    device = torch.device("cuda")
    records = get_records()
    train_indices, valid_indices = get_indices(records, args.fold)
    train_dataset = DetectionDataset(train_indices, args.features, records, training=True)
    valid_dataset = DetectionDataset(valid_indices, args.features, records)
    loader_options = {
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
    generator = torch.Generator().manual_seed(args.seed + args.fold)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
        **loader_options,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        **loader_options,
    )
    counts = np.bincount(
        [records[index]["label"] for index in train_indices], minlength=40
    ).clip(1)
    weights = np.sqrt(counts.mean() / counts)
    weights = torch.tensor(weights / weights.mean(), dtype=torch.float32, device=device)
    model = ObjectTokenTransformer().to(device)
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.02)
    total_steps = len(train_loader) * args.epochs
    name = f"v110_object_token_transformer_f{args.fold}"
    destination = HERE / "outputs" / name
    destination.mkdir(parents=True, exist_ok=False)
    config = {
        **vars(args),
        "features": str(args.features),
        "name": name,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train_rows": len(train_indices),
        "valid_rows": len(valid_indices),
        "validation_users": FOLDS[args.fold],
        "architecture": "three-layer Transformer over 8x3 pretrained-CNN object tokens",
        "precision": "BF16 autocast; FP32 optimizer and EMA",
    }
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    import swanlab

    run = swanlab.init(
        project="cuhkx-small-multimodal",
        name=name,
        config=config,
        public=False,
        mode=args.swanlab_mode,
        log_dir=str(HERE / "swanlog"),
        description="Object and scene token Transformer from the existing compact YOLO asset.",
    )
    global_step = 0
    started = time.monotonic()
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        correct = 0
        seen = 0
        loss_sum = 0.0
        for batch in train_loader:
            progress = global_step / max(total_steps - 1, 1)
            rate = 0.001 * 0.5 * (1 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = rate
            features = batch["features"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(features)
                loss = functional.cross_entropy(
                    logits, labels, weight=weights, label_smoothing=0.03
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            with torch.no_grad():
                decay = min(0.995, 1 - 1 / (global_step + 1))
                for target, source in zip(ema.parameters(), model.parameters()):
                    target.lerp_(source, 1 - decay)
            count = len(labels)
            loss_sum += float(loss.detach()) * count
            correct += int((logits.argmax(1) == labels).sum())
            seen += count
            global_step += 1
        arrays = evaluate(ema, valid_loader, device)
        valid_correct = int((arrays["probabilities"].argmax(1) == arrays["labels"]).sum())
        values = {
            "epoch": epoch,
            "train/loss": loss_sum / seen,
            "train/accuracy": correct / seen,
            "valid/correct": valid_correct,
            "valid/accuracy": valid_correct / len(arrays["labels"]),
            "train/lr": rate,
        }
        history.append(values)
        swanlab.log({key: value for key, value in values.items() if key != "epoch"}, step=epoch)
        print(json.dumps(values), flush=True)

    arrays = evaluate(ema, valid_loader, device)
    checkpoint = {
        "format": "v110_object_token_transformer_v1",
        "model": {
            name: value.detach().cpu().half() if value.is_floating_point() else value.detach().cpu()
            for name, value in ema.state_dict().items()
        },
        "config": config,
    }
    torch.save(checkpoint, destination / "model.pt")
    np.savez(destination / "validation.npz", **arrays)
    prediction = arrays["probabilities"].argmax(1)
    metrics = {
        "rows": len(prediction),
        "correct": int((prediction == arrays["labels"]).sum()),
        "accuracy": float((prediction == arrays["labels"]).mean()),
        "elapsed_seconds": time.monotonic() - started,
    }
    try:
        metrics["swanlab_url"] = run.url
    except ValueError:
        metrics["swanlab_url"] = None
    (destination / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (destination / "history.jsonl").write_text(
        "".join(json.dumps(values) + "\n" for values in history)
    )
    swanlab.finish()
    print("DONE " + json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()

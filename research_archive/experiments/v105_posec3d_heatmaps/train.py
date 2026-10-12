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

from experiments.v105_posec3d_heatmaps.model import PoseHeatmapCNN
from src.data import FOLDS, get_indices, get_records
from src.r2plus1d_models import compact_quantized_state_dict, quantize_state_dict


class HeatmapDataset(Dataset):
    def __init__(self, indices, cache_dir, training=False):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.cache_dir = Path(cache_dir)
        self.training = training
        self.records = get_records()
        self.manifest = json.loads((self.cache_dir / "manifest.json").read_text())
        self.shape = tuple(self.manifest["shape"])
        self.cache_path = str(self.cache_dir / "heatmaps.uint8.memmap")
        if Path(self.cache_path).stat().st_size != int(np.prod(self.shape)):
            raise ValueError("Incomplete heatmap cache")
        self.memmap = None
        self.channel_swap = torch.tensor([0, 2, 1, 4, 3] * 3)

    @property
    def cache(self):
        if self.memmap is None:
            self.memmap = np.memmap(self.cache_path, dtype=np.uint8, mode="r", shape=self.shape)
        return self.memmap

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = int(self.indices[position])
        values = torch.from_numpy(np.asarray(self.cache[index]).copy()).float().div_(255).permute(1, 0, 2, 3)
        if self.training:
            if random.random() < 0.5:
                values = values[self.channel_swap]
                values[:5] = torch.flip(values[:5], dims=(-1,))
                values[10:15] = torch.flip(values[10:15], dims=(-1,))
            if random.random() < 0.25:
                group = random.randrange(5)
                values[group::5] = 0
            shift_y = random.randint(-2, 2)
            shift_x = random.randint(-2, 2)
            values = torch.roll(values, shifts=(shift_y, shift_x), dims=(-2, -1))
            if shift_y > 0:
                values[..., :shift_y, :] = 0
            elif shift_y < 0:
                values[..., shift_y:, :] = 0
            if shift_x > 0:
                values[..., :shift_x] = 0
            elif shift_x < 0:
                values[..., shift_x:] = 0
        return {"heatmaps": values, "label": int(self.records[index]["label"]), "index": index}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=HERE / "outputs")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--warmup-epochs", type=float, default=2.0)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=42)
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
    return maximum * 0.5 * (1 + math.cos(math.pi * min(max(progress, 0), 1)))


def mixed_loss(logits, labels, mix_labels, coefficient, smoothing):
    first = functional.cross_entropy(logits, labels, label_smoothing=smoothing)
    second = functional.cross_entropy(logits, mix_labels, label_smoothing=smoothing)
    return coefficient * first + (1 - coefficient) * second


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    probabilities = []
    labels = []
    indices = []
    for batch in loader:
        values = batch["heatmaps"].cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(values)
        probabilities.append(logits.float().softmax(dim=1).cpu())
        labels.append(batch["label"])
        indices.append(batch["index"])
    probabilities = torch.cat(probabilities).numpy()
    labels = torch.cat(labels).numpy()
    indices = torch.cat(indices).numpy()
    prediction = probabilities.argmax(axis=1)
    return {
        "rows": len(labels),
        "correct": int((prediction == labels).sum()),
        "accuracy": float((prediction == labels).mean()),
        "nll": float(-np.log(probabilities[np.arange(len(labels)), labels].clip(1e-9)).mean()),
    }, {"probabilities": probabilities, "labels": labels, "indices": indices}


def main():
    args = parse_args()
    seed_everything(args.seed + args.fold)
    records = get_records()
    train_indices, valid_indices = get_indices(records, args.fold)
    train_loader = DataLoader(
        HeatmapDataset(train_indices, args.cache_dir, training=True),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed + args.fold),
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    valid_loader = DataLoader(
        HeatmapDataset(valid_indices, args.cache_dir),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    model = PoseHeatmapCNN().cuda()
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = round(len(train_loader) * args.warmup_epochs)
    destination = args.output_root / args.name
    destination.mkdir(parents=True, exist_ok=False)
    config = {
        **vars(args),
        "cache_dir": str(args.cache_dir),
        "output_root": str(args.output_root),
        "parameters": model.parameter_count(),
        "train_clips": len(train_indices),
        "valid_clips": len(valid_indices),
        "validation_users": FOLDS[args.fold],
        "architecture": "canonical three-view grouped-pose heatmaps plus depthwise residual 3D CNN",
        "precision": "BF16 autocast; FP32 optimizer and EMA",
        "cache_manifest_sha256": hashlib.sha256((args.cache_dir / "manifest.json").read_bytes()).hexdigest(),
    }
    serializable = {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}
    (destination / "config.json").write_text(json.dumps(serializable, indent=2) + "\n")

    import swanlab

    run = swanlab.init(
        project="cuhkx-small-multimodal",
        name=args.name,
        config=serializable,
        public=False,
        mode=args.swanlab_mode,
        log_dir=str(HERE / "swanlog"),
        description="PoseC3D-style canonical skeleton heatmap ablation.",
    )
    started = time.monotonic()
    global_step = 0
    best = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        correct = 0
        epoch_started = time.monotonic()
        for batch in train_loader:
            values = batch["heatmaps"].cuda(non_blocking=True)
            labels = batch["label"].cuda(non_blocking=True)
            coefficient = float(np.random.beta(args.mixup_alpha, args.mixup_alpha))
            permutation = torch.randperm(len(values), device=values.device)
            mixed = coefficient * values + (1 - coefficient) * values[permutation]
            rate = learning_rate(global_step, total_steps, warmup_steps, args.lr)
            for group in optimizer.param_groups:
                group["lr"] = rate
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(mixed)
                loss = mixed_loss(logits, labels, labels[permutation], coefficient, args.label_smoothing)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite V105 loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            with torch.no_grad():
                decay = min(args.ema_decay, 1 - 1 / (global_step + 1))
                for ema_parameter, parameter in zip(ema.parameters(), model.parameters()):
                    ema_parameter.lerp_(parameter, 1 - decay)
            count = len(labels)
            seen += count
            correct += int((logits.argmax(dim=1) == labels).sum())
            loss_sum += float(loss.detach()) * count
            global_step += 1
        metrics, arrays = evaluate(ema, valid_loader)
        values = {
            "epoch": epoch,
            "train/loss": loss_sum / seen,
            "train/accuracy": correct / seen,
            "train/lr": rate,
            "valid/accuracy": metrics["accuracy"],
            "valid/correct": metrics["correct"],
            "valid/nll": metrics["nll"],
            "system/epoch_seconds": time.monotonic() - epoch_started,
            "system/peak_cuda_mb": torch.cuda.max_memory_allocated() / 1e6,
        }
        with (destination / "history.jsonl").open("a") as handle:
            handle.write(json.dumps(values) + "\n")
        print(json.dumps(values), flush=True)
        swanlab.log({key: value for key, value in values.items() if key != "epoch"}, step=epoch)
        key = (metrics["correct"], -metrics["nll"], -epoch)
        if best is None or key > best[0]:
            state = {
                name: value.detach().cpu().half() if value.is_floating_point() else value.detach().cpu()
                for name, value in ema.state_dict().items()
            }
            torch.save({"model": state, "config": serializable, "epoch": epoch, "metrics": metrics}, destination / "best.pt")
            np.savez(destination / "validation.npz", **arrays)
            best = (key, epoch, metrics)

    checkpoint = torch.load(destination / "best.pt", map_location="cpu", weights_only=False)
    quantized, tensor_bytes = quantize_state_dict(checkpoint["model"])
    torch.save(
        {
            "model_int8": compact_quantized_state_dict(quantized),
            "config": serializable,
            "epoch": checkpoint["epoch"],
            "metrics": checkpoint["metrics"],
            "tensor_bytes": tensor_bytes,
        },
        destination / "best_int8.pt",
    )
    completion = {
        "status": "complete",
        "best_epoch": best[1],
        "metrics": best[2],
        "parameters": serializable["parameters"],
        "checkpoint_bytes": (destination / "best.pt").stat().st_size,
        "int8_checkpoint_bytes": (destination / "best_int8.pt").stat().st_size,
        "peak_cuda_mb": torch.cuda.max_memory_allocated() / 1e6,
        "elapsed_seconds": time.monotonic() - started,
    }
    try:
        completion["swanlab_url"] = run.url
    except ValueError:
        completion["swanlab_url"] = None
    (destination / "completion.json").write_text(json.dumps(completion, indent=2) + "\n")
    swanlab.finish()
    print("DONE " + json.dumps(completion), flush=True)


if __name__ == "__main__":
    main()

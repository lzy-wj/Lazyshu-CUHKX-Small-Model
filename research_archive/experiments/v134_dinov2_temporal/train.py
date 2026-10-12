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
from torch.nn import functional
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.v134_dinov2_temporal.model import DINOv2TemporalMultimodal
from src.data import FOLDS, get_indices, get_records


class FeatureDataset(Dataset):
    def __init__(self, indices, training=False):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.training = training
        manifest = json.loads((HERE / "features/manifest.json").read_text())
        self.visual = np.memmap(
            HERE / "features/dinov2_vits14_cls_patchmean.float16.memmap",
            dtype=np.float16,
            mode="r",
            shape=tuple(manifest["shape"]),
        )
        self.visual_mask = np.load(HERE / "features/visual_mask.npy", mmap_mode="r")
        self.skeleton = np.load(ROOT / "data/processed/skeleton.npy", mmap_mode="r")
        self.imu = np.load(ROOT / "data/processed/imu_world.npy", mmap_mode="r")
        self.sensor_mask = np.load(ROOT / "data/processed/sensor_mask.npy", mmap_mode="r")
        self.records = get_records()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = int(self.indices[position])
        visual = torch.from_numpy(np.asarray(self.visual[index]).astype(np.float32))
        if self.training:
            visual = visual + torch.randn_like(visual) * 0.01
        return {
            "visual": visual,
            "skeleton": torch.from_numpy(np.asarray(self.skeleton[index]).copy()),
            "imu": torch.from_numpy(np.asarray(self.imu[index]).copy()),
            "visual_mask": torch.from_numpy(np.asarray(self.visual_mask[index]).copy()),
            "sensor_mask": torch.from_numpy(np.asarray(self.sensor_mask[index, :2]).copy()),
            "label": self.records[index]["label"],
            "user": self.records[index]["user"],
            "index": index,
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--mixup", type=float, default=0.2)
    parser.add_argument("--contrastive-weight", type=float, default=0.0)
    parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--swanlab-mode", choices=("online", "offline", "disabled"), default="online")
    return parser.parse_args()


def to_device(batch, device, training=False):
    result = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    if training:
        result["skeleton"] = result["skeleton"] + torch.randn_like(result["skeleton"]) * 0.006
        result["imu"] = result["imu"] + torch.randn_like(result["imu"]) * 0.006
        availability = torch.cat([result["visual_mask"], result["sensor_mask"]], dim=1)
        dropped = availability * (torch.rand_like(availability) > 0.1)
        empty = dropped.sum(dim=1) == 0
        dropped[empty] = availability[empty]
        result["visual_mask"] = dropped[:, :3]
        result["sensor_mask"] = dropped[:, 3:]
    return result


def mixup(batch, alpha):
    if alpha <= 0:
        return None
    coefficient = float(np.random.beta(alpha, alpha))
    permutation = torch.randperm(len(batch["label"]), device=batch["label"].device)
    for name in ("visual", "skeleton", "imu"):
        batch[name] = coefficient * batch[name] + (1 - coefficient) * batch[name][permutation]
    batch["visual_mask"] = torch.maximum(batch["visual_mask"], batch["visual_mask"][permutation])
    batch["sensor_mask"] = torch.maximum(batch["sensor_mask"], batch["sensor_mask"][permutation])
    return permutation, coefficient


def cross_user_contrastive(embedding, labels, users, temperature):
    embedding = functional.normalize(embedding.float(), dim=1)
    logits = embedding @ embedding.t() / temperature
    identity = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positives = (labels[:, None] == labels[None, :]) & (users[:, None] != users[None, :])
    logits = logits.masked_fill(identity, -torch.inf)
    active = positives.any(dim=1)
    if not active.any():
        return embedding.sum() * 0
    log_probabilities = logits[active] - torch.logsumexp(logits[active], dim=1, keepdim=True)
    selected = positives[active]
    return -(log_probabilities.masked_fill(~selected, 0).sum(dim=1) / selected.sum(dim=1)).mean()


def model_inputs(batch):
    return {key: batch[key] for key in ("visual", "skeleton", "imu", "visual_mask", "sensor_mask")}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    probabilities, labels, indices = [], [], []
    for raw_batch in loader:
        batch = to_device(raw_batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(**model_inputs(batch))
        probabilities.append(logits.float().softmax(dim=1).cpu())
        labels.append(batch["label"].cpu())
        indices.append(batch["index"].cpu())
    probabilities = torch.cat(probabilities).numpy()
    labels = torch.cat(labels).numpy()
    indices = torch.cat(indices).numpy()
    return {
        "accuracy": float((probabilities.argmax(axis=1) == labels).mean()),
        "nll": float(-np.log(probabilities[np.arange(len(labels)), labels].clip(1e-9)).mean()),
    }, {"probabilities": probabilities, "labels": labels, "indices": indices}


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    random.seed(args.seed + args.fold)
    np.random.seed(args.seed + args.fold)
    torch.manual_seed(args.seed + args.fold)
    torch.cuda.manual_seed_all(args.seed + args.fold)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    records = get_records()
    train_indices, valid_indices = get_indices(records, args.fold)
    loader_options = {
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
    generator = torch.Generator().manual_seed(args.seed + args.fold)
    train_loader = DataLoader(
        FeatureDataset(train_indices, training=True),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
        **loader_options,
    )
    valid_loader = DataLoader(
        FeatureDataset(valid_indices), batch_size=args.batch_size, shuffle=False, **loader_options
    )
    model = DINOv2TemporalMultimodal().to(device)
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    destination = HERE / "outputs" / args.name
    destination.mkdir(parents=True, exist_ok=False)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    config = vars(args) | {
        "parameters": parameters,
        "dinov2_parameters": 22056576,
        "total_deployment_parameters": parameters + 22056576,
        "validation_users": FOLDS[args.fold],
        "train_clips": len(train_indices),
        "valid_clips": len(valid_indices),
        "architecture": "shared frozen DINOv2 ViT-S frame encoder + temporal Transformer + Skeleton CNN + IMU CNN/BiLSTM + fusion Transformer",
    }
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    import swanlab

    run = swanlab.init(
        project="cuhkx-small-multimodal",
        name=args.name,
        config=config,
        mode=args.swanlab_mode,
        public=False,
        log_dir=str(HERE / "swanlog"),
        description="Frozen DINOv2 frame semantics with temporal and multimodal Transformers.",
    )
    best = None
    stale = 0
    history = []
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        learning_rate = args.lr * 0.5 * (1 + math.cos(math.pi * (epoch - 1) / args.epochs))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        total_loss = total_correct = total_rows = 0
        for raw_batch in train_loader:
            batch = to_device(raw_batch, device, training=True)
            original_labels = batch["label"].clone()
            mixed = mixup(batch, args.mixup)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, details = model(**model_inputs(batch))
                primary = functional.cross_entropy(logits, original_labels, label_smoothing=0.03)
                auxiliary = functional.cross_entropy(
                    details["auxiliary"].flatten(0, 1),
                    original_labels[:, None].expand(-1, 5).reshape(-1),
                    label_smoothing=0.03,
                )
                if mixed is not None:
                    permutation, coefficient = mixed
                    primary = coefficient * primary + (1 - coefficient) * functional.cross_entropy(
                        logits, original_labels[permutation], label_smoothing=0.03
                    )
                    auxiliary = coefficient * auxiliary + (1 - coefficient) * functional.cross_entropy(
                        details["auxiliary"].flatten(0, 1),
                        original_labels[permutation][:, None].expand(-1, 5).reshape(-1),
                        label_smoothing=0.03,
                    )
                contrastive = cross_user_contrastive(
                    details["embedding"], original_labels, batch["user"], args.contrastive_temperature
                )
                loss = primary + 0.15 * auxiliary + args.contrastive_weight * contrastive
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            with torch.no_grad():
                for target, source in zip(ema.parameters(), model.parameters()):
                    target.lerp_(source, 1 - 0.98)
            total_loss += float(loss.detach()) * len(original_labels)
            total_correct += int((logits.argmax(dim=1) == original_labels).sum())
            total_rows += len(original_labels)
        metrics, arrays = evaluate(ema, valid_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / total_rows,
            "train_accuracy": total_correct / total_rows,
            "valid_accuracy": metrics["accuracy"],
            "valid_nll": metrics["nll"],
            "lr": learning_rate,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        swanlab.log(row, step=epoch)
        key = (metrics["accuracy"], -metrics["nll"])
        if best is None or key > best[0]:
            best = (key, epoch, copy.deepcopy(ema.state_dict()), metrics, arrays)
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    _, best_epoch, state, metrics, arrays = best
    torch.save({"model": state, "config": config, "epoch": best_epoch, "metrics": metrics}, destination / "best.pt")
    np.savez(destination / "validation.npz", **arrays)
    (destination / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    completion = {
        "status": "complete",
        "best_epoch": best_epoch,
        "metrics": metrics,
        "parameters": parameters,
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

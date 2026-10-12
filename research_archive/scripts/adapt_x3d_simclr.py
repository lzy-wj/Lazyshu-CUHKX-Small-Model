#!/usr/bin/env python3
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ROOT, ClipDataset, get_indices, get_records, prepare_batch
from src.models import MultiModalModel


def contrastive_loss(left, right, temperature):
    left = functional.normalize(left, dim=1)
    right = functional.normalize(right, dim=1)
    values = torch.cat([left, right], dim=0)
    logits = values @ values.T / temperature
    logits.fill_diagonal_(torch.finfo(logits.dtype).min)
    count = len(left)
    labels = torch.cat(
        [torch.arange(count, 2 * count, device=values.device), torch.arange(count, device=values.device)]
    )
    return functional.cross_entropy(logits, labels)


def evaluate(model, loader, device):
    model.eval()
    probabilities, labels, indices = [], [], []
    with torch.inference_mode():
        for raw_batch in loader:
            batch = prepare_batch(raw_batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(batch)
            probabilities.append(logits.float().softmax(1).cpu().numpy())
            labels.append(batch["label"].cpu().numpy())
            indices.append(batch["index"].cpu().numpy())
    probabilities = np.concatenate(probabilities)
    labels = np.concatenate(labels)
    indices = np.concatenate(indices)
    return float((probabilities.argmax(1) == labels).mean()), {
        "probabilities": probabilities,
        "labels": labels,
        "indices": indices,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr-factor", type=float, default=0.1)
    parser.add_argument("--contrastive-weight", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--swanlab-mode", choices=("online", "offline", "disabled"), default="online")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = source["config"]
    if int(config["fold"]) != args.fold:
        raise ValueError("Checkpoint fold mismatch")
    if config.get("backbone") != "x3d_s":
        raise ValueError("SimCLR adaptation currently supports X3D checkpoints")
    model = MultiModalModel(
        config["modalities"],
        width=config["width"],
        spatial_pool=config.get("spatial_pool", 1),
        backbone=config["backbone"],
        skeleton_backbone=config.get("skeleton_backbone", "tcn"),
        temporal_backbone=config.get("temporal_backbone", "tcn"),
        temporal_shift=config.get("temporal_shift", False),
        kinetics_source=config.get("kinetics_source", "s"),
        fusion=config.get("fusion", "concat"),
    ).to(device)
    model.load_state_dict(source["model"], strict=True)
    projector = nn.Sequential(
        nn.Linear(384, 384), nn.BatchNorm1d(384), nn.SiLU(), nn.Linear(384, 128)
    ).to(device)
    records = get_records()
    train_indices, target_indices = get_indices(records, args.fold)
    dataset_options = {
        "frames": config["frames"],
        "canonical_pose": config.get("canonical_pose", False),
        "image_size": config.get("image_size", 96),
        "thermal_view": config.get("thermal_view", "full"),
        "imu_representation": config.get("imu_representation", "raw"),
    }
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
    supervised_loader = DataLoader(
        ClipDataset(train_indices, config["modalities"], training=True, **dataset_options),
        shuffle=True,
        drop_last=True,
        **loader_options,
    )
    target_loader = DataLoader(
        ClipDataset(target_indices, config["modalities"], training=True, **dataset_options),
        shuffle=True,
        drop_last=True,
        **loader_options,
    )
    validation_loader = DataLoader(
        ClipDataset(target_indices, config["modalities"], **dataset_options),
        shuffle=False,
        **loader_options,
    )
    backbone_parameters = [parameter for name, parameter in model.named_parameters() if ".frames." in name]
    head_parameters = [parameter for name, parameter in model.named_parameters() if ".frames." not in name]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": args.lr * args.backbone_lr_factor},
            {"params": head_parameters, "lr": args.lr},
            {"params": projector.parameters(), "lr": args.lr},
        ],
        weight_decay=1e-4,
    )
    class_counts = np.bincount([records[index]["label"] for index in train_indices], minlength=40).clip(1)
    class_weights = torch.tensor(
        (class_counts.mean() / class_counts) ** 0.25, dtype=torch.float32, device=device
    )
    destination = ROOT / "outputs/checkpoints" / args.name
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "best.pt").exists():
        raise RuntimeError("Adaptation output already exists")
    adaptation_config = vars(args).copy()
    adaptation_config.update(
        {
            "checkpoint": str(args.checkpoint),
            "source_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            "source_name": config["name"],
            "modalities": config["modalities"],
            "target_clips": len(target_indices),
            "target_label_policy": "Target labels are excluded from optimization and used only for fold validation.",
            "architecture": "X3D CNN encoders + projection MLP; supervised CE plus target-domain NT-Xent",
            "precision": "BF16 autocast; FP32 optimizer",
        }
    )
    (destination / "config.json").write_text(json.dumps(adaptation_config, indent=2, default=str) + "\n")
    (destination / "split.json").write_text(json.dumps({"train": train_indices, "target": target_indices}) + "\n")
    import swanlab
    run = swanlab.init(
        project="cuhkx-small-multimodal",
        name=args.name,
        config=adaptation_config,
        public=False,
        mode=args.swanlab_mode,
        log_dir=str(ROOT / "outputs/swanlog"),
        description="Unlabeled target-domain SimCLR with labeled source classification; target labels are validation only.",
    )
    best_accuracy = -1.0
    best_epoch = 0
    target_iterator = iter(target_loader)
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        projector.train()
        for name, layer in model.named_modules():
            if ".frames." in name and isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
                layer.eval()
        supervised_iterator = iter(supervised_loader)
        steps = args.steps_per_epoch or len(supervised_loader)
        supervised_total = 0.0
        contrastive_total = 0.0
        for _ in range(steps):
            try:
                source_batch = next(supervised_iterator)
            except StopIteration:
                supervised_iterator = iter(supervised_loader)
                source_batch = next(supervised_iterator)
            try:
                target_batch = next(target_iterator)
            except StopIteration:
                target_iterator = iter(target_loader)
                target_batch = next(target_iterator)
            optimizer.zero_grad(set_to_none=True)
            source_values = prepare_batch(source_batch, device, training=True, strong_augment=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                source_logits, auxiliary = model(source_values)
                supervised = functional.cross_entropy(
                    source_logits, source_values["label"], weight=class_weights, label_smoothing=0.05
                )
                for name, logits in auxiliary.items():
                    losses = functional.cross_entropy(logits, source_values["label"], reduction="none", label_smoothing=0.05)
                    mask = source_values[name + "_mask"]
                    supervised = supervised + 0.15 / len(auxiliary) * (losses * mask).sum() / mask.sum().clamp_min(1)
            supervised.backward()
            left = prepare_batch(target_batch, device, training=True, strong_augment=True, spatial_augment=True)
            right = prepare_batch(target_batch, device, training=True, strong_augment=True, spatial_augment=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, _, left_embedding = model(left, return_embedding=True)
                _, _, right_embedding = model(right, return_embedding=True)
                contrastive = contrastive_loss(
                    projector(left_embedding).float(), projector(right_embedding).float(), args.temperature
                )
            (args.contrastive_weight * contrastive).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            supervised_total += float(supervised.detach())
            contrastive_total += float(contrastive.detach())
        accuracy, predictions = evaluate(model, validation_loader, device)
        metrics = {
            "train/supervised_loss": supervised_total / steps,
            "train/contrastive_loss": contrastive_total / steps,
            "valid/accuracy": accuracy,
        }
        swanlab.log(metrics, step=epoch)
        with (destination / "history.jsonl").open("a") as handle:
            handle.write(json.dumps({"epoch": epoch} | metrics) + "\n")
        print(f"epoch={epoch} " + " ".join(f"{key}={value:.5f}" for key, value in metrics.items()), flush=True)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
            torch.save(
                {
                    "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                    "config": config | {"name": args.name, "adaptation": adaptation_config},
                    "epoch": epoch,
                    "metrics": {"accuracy": accuracy},
                },
                destination / "best.pt",
            )
            np.savez(destination / "validation.npz", **predictions)
            (destination / "metrics.json").write_text(
                json.dumps({"accuracy": accuracy, "epoch": epoch}, indent=2) + "\n"
            )
    print(
        f"DONE best_accuracy={best_accuracy:.5f} best_epoch={best_epoch} "
        f"minutes={(time.monotonic() - started) / 60:.2f}",
        flush=True,
    )
    swanlab.finish()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.v95_phase_aligned_label_experts.model import LABEL_GROUPS, build_from_source
from scripts.train_r2plus1d import mixed_cross_entropy, mixup_batch, prepare_batch, seed_everything
from src.data import FOLDS, get_indices, get_records
from src.r2plus1d_data import R2Plus1DDataset
from src.r2plus1d_models import compact_quantized_state_dict, quantize_state_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=HERE / "outputs")
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--dimension", type=int, default=384)
    parser.add_argument("--phase-depth", type=int, default=1)
    parser.add_argument("--expert-depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--feedforward-multiplier", type=int, default=3)
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument("--phases", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--mixup-alpha", type=float, default=0.3)
    parser.add_argument("--expert-loss-weight", type=float, default=0.25)
    parser.add_argument("--retention-weight", type=float, default=0.5)
    parser.add_argument("--retention-temperature", type=float, default=2.0)
    parser.add_argument("--limit-train-batches", type=int, default=0)
    parser.add_argument("--limit-valid-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--swanlab-mode", choices=("cloud", "local", "disabled"), default="cloud")
    return parser.parse_args()


def model_inputs(batch):
    return {
        "video": batch["video"],
        "skeleton": batch["skeleton"],
        "imu": batch["imu"],
        "sensor_mask": batch["sensor_mask"],
        "visual_mask": batch["visual_mask"],
    }


def mapped_labels(labels, group):
    mapping = torch.full((40,), len(group), dtype=torch.long, device=labels.device)
    group_indices = torch.tensor(group, dtype=torch.long, device=labels.device)
    mapping[group_indices] = torch.arange(len(group), device=labels.device)
    return mapping[labels]


def expert_loss(expert_logits, batch, label_smoothing):
    losses = []
    for group, logits in zip(LABEL_GROUPS, expert_logits):
        primary = functional.cross_entropy(
            logits,
            mapped_labels(batch["label"], group),
            label_smoothing=label_smoothing,
        )
        if "mix_label" in batch:
            secondary = functional.cross_entropy(
                logits,
                mapped_labels(batch["mix_label"], group),
                label_smoothing=label_smoothing,
            )
            primary = batch["mix_lambda"] * primary + (1 - batch["mix_lambda"]) * secondary
        losses.append(primary)
    return torch.stack(losses).mean()


def learning_rate(step, total_steps, warmup_steps, maximum):
    if warmup_steps and step < warmup_steps:
        return maximum * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return maximum * 0.5 * (1 + math.cos(math.pi * min(max(progress, 0), 1)))


@torch.no_grad()
def evaluate(model, loader, device, limit_batches=0):
    model.eval()
    probabilities = []
    labels = []
    indices = []
    for batch_number, raw_batch in enumerate(loader):
        batch = prepare_batch(raw_batch, device, multimodal=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(**model_inputs(batch))
        probabilities.append(logits.float().softmax(dim=1).cpu())
        labels.append(batch["label"].cpu())
        indices.append(batch["index"].cpu())
        if limit_batches and batch_number + 1 >= limit_batches:
            break
    probabilities = torch.cat(probabilities).numpy()
    labels = torch.cat(labels).numpy()
    indices = torch.cat(indices).numpy()
    prediction = probabilities.argmax(axis=1)
    records = get_records()
    users = np.asarray([records[int(index)]["user"] for index in indices])
    return (
        {
            "rows": len(labels),
            "correct": int((prediction == labels).sum()),
            "accuracy": float((prediction == labels).mean()),
            "nll": float(-np.log(probabilities[np.arange(len(labels)), labels].clip(1e-9)).mean()),
            "per_user": {
                str(user): {
                    "rows": int((users == user).sum()),
                    "correct": int(((prediction == labels) & (users == user)).sum()),
                    "accuracy": float((prediction[users == user] == labels[users == user]).mean()),
                }
                for user in np.unique(users)
            },
        },
        {"probabilities": probabilities, "labels": labels, "indices": indices, "users": users},
    )


def main():
    args = parse_args()
    if args.retention_temperature <= 0:
        raise ValueError("Retention temperature must be positive")
    seed_everything(args.seed + args.fold)
    device = torch.device("cuda")
    model, source_config = build_from_source(
        args.init_checkpoint,
        dimension=args.dimension,
        phase_depth=args.phase_depth,
        expert_depth=args.expert_depth,
        heads=args.heads,
        feedforward_multiplier=args.feedforward_multiplier,
        residual_scale=args.residual_scale,
        phases=args.phases,
    )
    if int(source_config["fold"]) != args.fold:
        raise ValueError("Initialization checkpoint and validation fold disagree")
    model.to(device)
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)

    records = get_records()
    train_indices, valid_indices = get_indices(records, args.fold)
    dataset_options = {
        "cache_dir": args.cache_dir,
        "include_sensors": True,
        "include_radar": False,
        "frames": args.frames,
        "image_size": args.image_size,
    }
    loader_options = {
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(
        R2Plus1DDataset(
            train_indices,
            training=True,
            horizontal_flip_probability=float(source_config.get("horizontal_flip_probability", 0.5)),
            **dataset_options,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed + args.fold),
        **loader_options,
    )
    valid_loader = DataLoader(
        R2Plus1DDataset(valid_indices, training=False, **dataset_options),
        batch_size=args.batch_size,
        shuffle=False,
        **loader_options,
    )
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_steps = len(train_loader) * args.epochs
    warmup_steps = round(len(train_loader) * args.warmup_epochs)
    destination = args.output_root / args.name
    destination.mkdir(parents=True, exist_ok=False)
    config = {
        **vars(args),
        "init_checkpoint": str(args.init_checkpoint),
        "cache_dir": str(args.cache_dir),
        "output_root": str(args.output_root),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "adapter_parameters": model.adapter_parameter_count(),
        "trainable_parameters": sum(parameter.numel() for _, parameter in trainable),
        "validation_users": FOLDS[args.fold],
        "label_groups": [list(group) for group in LABEL_GROUPS],
        "source_checkpoint_sha256": hashlib.sha256(args.init_checkpoint.read_bytes()).hexdigest(),
        "architecture": "frozen R(2+1)D/TCN/LSTM encoders + phase Transformer + four label-expert temporal Transformers",
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
        description="Phase-aligned multimodal teacher with fixed semantic label experts.",
    )
    global_step = 0
    started = time.monotonic()
    history = []
    trainable_names = {name for name, _ in trainable}
    ema_parameters = dict(ema.named_parameters())
    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = {"loss": 0.0, "classification": 0.0, "expert": 0.0, "retention": 0.0}
        seen = 0
        correct = 0
        epoch_started = time.monotonic()
        for batch_number, raw_batch in enumerate(train_loader):
            batch = prepare_batch(raw_batch, device, multimodal=True, training=True)
            mixup_batch(batch, args.mixup_alpha)
            rate = learning_rate(global_step, total_steps, warmup_steps, args.lr)
            for group in optimizer.param_groups:
                group["lr"] = rate
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, details = model(**model_inputs(batch))
                classification = mixed_cross_entropy(
                    logits,
                    batch,
                    label_smoothing=args.label_smoothing,
                )
                specialists = expert_loss(
                    details["expert_logits"],
                    batch,
                    args.label_smoothing,
                )
                temperature = args.retention_temperature
                retention = functional.kl_div(
                    functional.log_softmax(logits.float() / temperature, dim=1),
                    functional.softmax(details["base_logits"].float() / temperature, dim=1),
                    reduction="batchmean",
                ) * temperature ** 2
                loss = (
                    classification
                    + args.expert_loss_weight * specialists
                    + args.retention_weight * retention
                )
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite V95 loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([parameter for _, parameter in trainable], 5)
            optimizer.step()
            with torch.no_grad():
                decay = min(args.ema_decay, 1 - 1 / (global_step + 1))
                for name, parameter in model.named_parameters():
                    if name in trainable_names:
                        ema_parameters[name].lerp_(parameter, 1 - decay)
            count = len(batch["label"])
            sums["loss"] += float(loss.detach()) * count
            sums["classification"] += float(classification.detach()) * count
            sums["expert"] += float(specialists.detach()) * count
            sums["retention"] += float(retention.detach()) * count
            correct += int((logits.argmax(dim=1) == batch["label"]).sum())
            seen += count
            global_step += 1
            if args.limit_train_batches and batch_number + 1 >= args.limit_train_batches:
                break
        values = {
            "epoch": epoch,
            "train/loss": sums["loss"] / seen,
            "train/classification": sums["classification"] / seen,
            "train/expert": sums["expert"] / seen,
            "train/retention": sums["retention"] / seen,
            "train/accuracy": correct / seen,
            "train/lr": rate,
            "system/epoch_seconds": time.monotonic() - epoch_started,
            "system/peak_cuda_mb": torch.cuda.max_memory_allocated() / 1e6,
        }
        history.append(values)
        with (destination / "history.jsonl").open("a") as handle:
            handle.write(json.dumps(values) + "\n")
        print(json.dumps(values), flush=True)
        swanlab.log({key: value for key, value in values.items() if key != "epoch"}, step=epoch)

    metrics, arrays = evaluate(ema, valid_loader, device, args.limit_valid_batches)
    checkpoint = {
        "model": {
            name: value.detach().cpu().half() if value.is_floating_point() else value.detach().cpu()
            for name, value in ema.state_dict().items()
        },
        "config": serializable,
        "epoch": args.epochs,
        "metrics": metrics,
    }
    torch.save(checkpoint, destination / "best.pt")
    np.savez(destination / "validation.npz", **arrays)
    (destination / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    quantized, tensor_bytes = quantize_state_dict(checkpoint["model"])
    torch.save(
        {
            "model_int8": compact_quantized_state_dict(quantized),
            "config": serializable,
            "metrics": metrics,
            "tensor_bytes": tensor_bytes,
        },
        destination / "best_int8.pt",
    )
    completion = {
        "status": "complete",
        "metrics": metrics,
        "parameters": serializable["parameters"],
        "adapter_parameters": serializable["adapter_parameters"],
        "checkpoint_bytes": (destination / "best.pt").stat().st_size,
        "int8_checkpoint_bytes": (destination / "best_int8.pt").stat().st_size,
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

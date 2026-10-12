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

from experiments.v96_modality_specialized_backbone.model import build_from_source as build_specialized
from scripts.train_r2plus1d import (
    mixed_cross_entropy,
    mixed_stream_cross_entropy,
    mixup_batch,
    prepare_batch,
    seed_everything,
)
from src.data import FOLDS, get_indices, get_records
from src.r2plus1d_data import R2Plus1DDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=HERE / "outputs")
    parser.add_argument(
        "--model-kind",
        choices=("specialized", "protected_adapter"),
        default="specialized",
    )
    parser.add_argument("--split-stage", type=int, choices=(3, 4), required=True)
    parser.add_argument("--bottleneck", type=int, default=128)
    parser.add_argument("--adapter-location", choices=("intermediate", "dual"), default="dual")
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--mixup-alpha", type=float, default=0.3)
    parser.add_argument("--auxiliary-weight", type=float, default=0.15)
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


def learning_rate(step, total_steps, warmup_steps, maximum):
    if warmup_steps and step < warmup_steps:
        return maximum * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return maximum * 0.5 * (1 + math.cos(math.pi * min(max(progress, 0), 1)))


def subset_metrics(mask, labels, predictions):
    rows = int(mask.sum())
    if rows == 0:
        return {"rows": 0, "correct": 0, "accuracy": None}
    correct = int((predictions[mask] == labels[mask]).sum())
    return {"rows": rows, "correct": correct, "accuracy": correct / rows}


@torch.no_grad()
def evaluate(model, loader, device, limit_batches=0):
    model.eval()
    probabilities = []
    labels = []
    indices = []
    visual_complete = []
    for batch_number, raw_batch in enumerate(loader):
        batch = prepare_batch(raw_batch, device, multimodal=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(**model_inputs(batch))
        probabilities.append(logits.float().softmax(dim=1).cpu())
        labels.append(batch["label"].cpu())
        indices.append(batch["index"].cpu())
        visual_complete.append(batch["visual_mask"].bool().all(dim=1).cpu())
        if limit_batches and batch_number + 1 >= limit_batches:
            break
    probabilities = torch.cat(probabilities).numpy()
    labels = torch.cat(labels).numpy()
    indices = torch.cat(indices).numpy()
    visual_complete = torch.cat(visual_complete).numpy().astype(bool)
    predictions = probabilities.argmax(axis=1)
    records = get_records()
    users = np.asarray([records[int(index)]["user"] for index in indices])
    report = {
        "rows": len(labels),
        "correct": int((predictions == labels).sum()),
        "accuracy": float((predictions == labels).mean()),
        "nll": float(-np.log(probabilities[np.arange(len(labels)), labels].clip(1e-9)).mean()),
        "visual_complete": subset_metrics(visual_complete, labels, predictions),
        "visual_missing": subset_metrics(~visual_complete, labels, predictions),
        "per_user": {
            str(user): subset_metrics(users == user, labels, predictions)
            for user in np.unique(users)
        },
    }
    report["worst_user_accuracy"] = min(value["accuracy"] for value in report["per_user"].values())
    arrays = {
        "probabilities": probabilities,
        "labels": labels,
        "indices": indices,
        "users": users,
    }
    return report, arrays


def update_ema(ema, model, trainable_names, decay):
    ema_parameters = dict(ema.named_parameters())
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in trainable_names:
                ema_parameters[name].lerp_(parameter, 1 - decay)
        for target, source in zip(ema.buffers(), model.buffers()):
            target.copy_(source)


def save_checkpoint(model, config, metrics, arrays, destination, epoch):
    state = {
        "model": {
            name: value.detach().cpu().half() if value.is_floating_point() else value.detach().cpu()
            for name, value in model.state_dict().items()
        },
        "config": config,
        "epoch": epoch,
        "metrics": metrics,
    }
    torch.save(state, destination / "best.pt")
    np.savez(destination / "validation.npz", **arrays)
    (destination / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


def main():
    args = parse_args()
    if args.retention_temperature <= 0:
        raise ValueError("Retention temperature must be positive")
    seed_everything(args.seed + args.fold)
    device = torch.device("cuda")
    if args.model_kind == "specialized":
        model, teacher, source_config = build_specialized(
            args.init_checkpoint,
            split_stage=args.split_stage,
        )
        architecture = (
            "shared R(2+1)D stem/low stages + modality-specialized high residual stages; "
            "frozen temporal sensor encoders and Transformer fusion"
        )
    else:
        from experiments.v97_protected_visual_adapters.model import build_from_source as build_adapter

        model, teacher, source_config = build_adapter(
            args.init_checkpoint,
            bottleneck=args.bottleneck,
            output_adapters=args.adapter_location == "dual",
        )
        architecture = (
            "frozen R(2+1)D backbone with zero-initialized modality-specific spatiotemporal "
            f"residual adapters at {args.adapter_location} stages"
        )
    if int(source_config["fold"]) != args.fold:
        raise ValueError("Initialization checkpoint and validation fold disagree")
    model.to(device)
    teacher.to(device)
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
    trainable_names = {name for name, _ in trainable}
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    destination = args.output_root / args.name
    destination.mkdir(parents=True, exist_ok=False)
    config = {
        **vars(args),
        "init_checkpoint": str(args.init_checkpoint),
        "cache_dir": str(args.cache_dir),
        "output_root": str(args.output_root),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for _, parameter in trainable),
        "validation_users": FOLDS[args.fold],
        "source_checkpoint_sha256": hashlib.sha256(args.init_checkpoint.read_bytes()).hexdigest(),
        "architecture": architecture,
        "selection": "fixed final epoch on both folds",
        "precision": "BF16 autocast; FP32 optimizer and EMA",
    }
    serializable = {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}
    (destination / "config.json").write_text(json.dumps(serializable, indent=2) + "\n")
    (destination / "split.json").write_text(json.dumps({"train": train_indices, "valid": valid_indices}) + "\n")

    import swanlab

    run = swanlab.init(
        project="cuhkx-small-multimodal",
        name=args.name,
        config=serializable,
        public=False,
        mode=args.swanlab_mode,
        log_dir=str(HERE / "swanlog"),
        description=architecture,
    )
    initial_metrics, _ = evaluate(ema, valid_loader, device, args.limit_valid_batches)
    print(
        json.dumps(
            {
                "epoch": 0,
                "valid/correct": initial_metrics["correct"],
                "valid/accuracy": initial_metrics["accuracy"],
                "valid/nll": initial_metrics["nll"],
            }
        ),
        flush=True,
    )
    swanlab.log(
        {
            "valid/accuracy": initial_metrics["accuracy"],
            "valid/correct": initial_metrics["correct"],
            "valid/nll": initial_metrics["nll"],
        },
        step=0,
    )

    total_steps = len(train_loader) * args.epochs
    warmup_steps = round(len(train_loader) * args.warmup_epochs)
    global_step = 0
    started = time.monotonic()
    final_metrics = None
    final_arrays = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = {"loss": 0.0, "classification": 0.0, "auxiliary": 0.0, "retention": 0.0}
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
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                teacher_logits, _ = teacher(**model_inputs(batch))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, details = model(**model_inputs(batch))
                classification = mixed_cross_entropy(
                    logits,
                    batch,
                    label_smoothing=args.label_smoothing,
                )
                stream_losses = mixed_stream_cross_entropy(
                    details["video_streams"],
                    batch,
                    label_smoothing=args.label_smoothing,
                )
                visual_mask = batch["visual_mask"].float()
                auxiliary = (stream_losses * visual_mask).sum() / visual_mask.sum().clamp_min(1)
                temperature = args.retention_temperature
                retention = functional.kl_div(
                    functional.log_softmax(logits.float() / temperature, dim=1),
                    functional.softmax(teacher_logits.float() / temperature, dim=1),
                    reduction="batchmean",
                ) * temperature ** 2
                loss = (
                    classification
                    + args.auxiliary_weight * auxiliary
                    + args.retention_weight * retention
                )
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite V96 loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([parameter for _, parameter in trainable], 5)
            optimizer.step()
            decay = min(args.ema_decay, 1 - 1 / (global_step + 1))
            update_ema(ema, model, trainable_names, decay)
            count = len(batch["label"])
            sums["loss"] += float(loss.detach()) * count
            sums["classification"] += float(classification.detach()) * count
            sums["auxiliary"] += float(auxiliary.detach()) * count
            sums["retention"] += float(retention.detach()) * count
            correct += int((logits.argmax(dim=1) == batch["label"]).sum())
            seen += count
            global_step += 1
            if args.limit_train_batches and batch_number + 1 >= args.limit_train_batches:
                break
        final_metrics, final_arrays = evaluate(ema, valid_loader, device, args.limit_valid_batches)
        values = {
            "epoch": epoch,
            "train/loss": sums["loss"] / seen,
            "train/classification": sums["classification"] / seen,
            "train/auxiliary": sums["auxiliary"] / seen,
            "train/retention": sums["retention"] / seen,
            "train/accuracy": correct / seen,
            "train/lr": rate,
            "valid/accuracy": final_metrics["accuracy"],
            "valid/correct": final_metrics["correct"],
            "valid/nll": final_metrics["nll"],
            "valid/worst_user_accuracy": final_metrics["worst_user_accuracy"],
            "system/epoch_seconds": time.monotonic() - epoch_started,
            "system/peak_cuda_mb": torch.cuda.max_memory_allocated() / 1e6,
        }
        with (destination / "history.jsonl").open("a") as handle:
            handle.write(json.dumps(values) + "\n")
        print(json.dumps(values), flush=True)
        swanlab.log({key: value for key, value in values.items() if key != "epoch"}, step=epoch)

    save_checkpoint(ema, serializable, final_metrics, final_arrays, destination, args.epochs)
    completion = {
        "status": "complete",
        "metrics": final_metrics,
        "initial_metrics": initial_metrics,
        "parameters": serializable["parameters"],
        "trainable_parameters": serializable["trainable_parameters"],
        "checkpoint_bytes": (destination / "best.pt").stat().st_size,
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

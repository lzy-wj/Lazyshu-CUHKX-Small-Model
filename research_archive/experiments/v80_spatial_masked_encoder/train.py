#!/usr/bin/env python3
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from experiments.v24_iterative_loop.train_iterative import (
    learning_rate,
    mixed_cross_entropy,
    mixup_batch,
    prepare_batch,
    selection_key,
    seed_everything,
    subset_metrics,
)
from model import build_from_source
from src.data import FOLDS, get_indices, get_records
from src.r2plus1d_data import R2Plus1DDataset
from src.r2plus1d_models import compact_quantized_state_dict, quantize_state_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1), default=0)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=HERE / "outputs")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--dimension", type=int, default=512)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--feedforward-multiplier", type=int, default=3)
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument("--reconstruction-mask-probability", type=float, default=0.35)
    parser.add_argument("--reconstruction-weight", type=float, default=0.2)
    parser.add_argument("--retention-weight", type=float, default=0.5)
    parser.add_argument("--retention-temperature", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--mixup-alpha", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train-batches", type=int, default=0)
    parser.add_argument("--limit-valid-batches", type=int, default=0)
    parser.add_argument("--select-final", action="store_true")
    parser.add_argument("--swanlab-mode", choices=("cloud", "local", "disabled"), default="cloud")
    return parser.parse_args()


def model_inputs(batch):
    return {
        "video": batch["video"],
        "skeleton": batch["skeleton"],
        "imu": batch["imu"],
        "radar": batch.get("radar"),
        "sensor_mask": batch["sensor_mask"],
        "visual_mask": batch["visual_mask"],
    }


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
    predictions = probabilities.argmax(1)
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
    arrays = {"probabilities": probabilities, "labels": labels, "indices": indices, "users": users}
    return report, arrays


def save_checkpoint(model, config, epoch, metrics, arrays, destination):
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
    seed_everything(args.seed + args.fold)
    device = torch.device("cuda")
    model, source_config = build_from_source(
        args.init_checkpoint,
        dimension=args.dimension,
        depth=args.depth,
        heads=args.heads,
        feedforward_multiplier=args.feedforward_multiplier,
        residual_scale=args.residual_scale,
        reconstruction_mask_probability=args.reconstruction_mask_probability,
    )
    model.to(device)
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)

    records = get_records()
    train_indices, valid_indices = get_indices(records, args.fold)
    cache_dir = args.cache_dir or Path(source_config["cache_dir"])
    image_size = args.image_size or int(source_config["image_size"])
    dataset_options = {
        "cache_dir": cache_dir,
        "include_sensors": True,
        "include_radar": False,
        "frames": int(source_config["frames"]),
        "image_size": image_size,
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
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    destination = args.output_root / args.name
    destination.mkdir(parents=True, exist_ok=False)
    config = {
        **source_config,
        **vars(args),
        "init_checkpoint": str(args.init_checkpoint),
        "output_root": str(args.output_root),
        "cache_dir": str(cache_dir),
        "image_size": image_size,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "inference_parameters": model.inference_parameter_count(),
        "training_only_parameters": sum(
            parameter.numel() for parameter in model.reconstruction_decoder.parameters()
        ),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "validation_users": FOLDS[args.fold],
        "source_checkpoint_sha256": hashlib.sha256(args.init_checkpoint.read_bytes()).hexdigest(),
        "architecture": "frozen R(2+1)D layer3 spatial grid + five-modality Transformer + training-only masked feature decoder",
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
        description="V80 spatially preserved CNN-Transformer encoder with masked feature reconstruction.",
    )
    best_metrics = None
    best_key = None
    best_epoch = 0
    if args.fold == 0:
        best_metrics, best_arrays = evaluate(model, valid_loader, device, args.limit_valid_batches)
        best_key = selection_key(best_metrics)
        if not args.select_final:
            save_checkpoint(model, serializable, 0, best_metrics, best_arrays, destination)
        print(json.dumps({"epoch": 0, "valid/correct": best_metrics["correct"], "valid/nll": best_metrics["nll"]}), flush=True)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = round(len(train_loader) * args.warmup_epochs)
    global_step = 0
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        seen = 0
        loss_sum = 0.0
        classification_sum = 0.0
        reconstruction_sum = 0.0
        retention_sum = 0.0
        correct = 0
        epoch_started = time.monotonic()
        for batch_number, raw_batch in enumerate(train_loader):
            rate = learning_rate(global_step, total_steps, warmup_steps, args.lr)
            for group in optimizer.param_groups:
                group["lr"] = rate
            batch = prepare_batch(raw_batch, device, multimodal=True, training=True)
            mixup_batch(batch, args.mixup_alpha)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, details = model(**model_inputs(batch), return_training_details=True)
                classification_loss = mixed_cross_entropy(
                    logits,
                    batch,
                    label_smoothing=args.label_smoothing,
                )
                temperature = args.retention_temperature
                retention_loss = functional.kl_div(
                    functional.log_softmax(logits.float() / temperature, dim=1),
                    functional.softmax(details["base_logits"].float() / temperature, dim=1),
                    reduction="batchmean",
                ) * temperature ** 2
                token_losses = functional.smooth_l1_loss(
                    details["reconstruction_predictions"].float(),
                    details["reconstruction_targets"].float(),
                    reduction="none",
                ).mean(dim=2)
                mask = details["reconstruction_mask"].float()
                reconstruction_loss = (token_losses * mask).sum() / mask.sum().clamp_min(1)
                loss = (
                    classification_loss
                    + args.retention_weight * retention_loss
                    + args.reconstruction_weight * reconstruction_loss
                )
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5)
            optimizer.step()
            with torch.no_grad():
                decay = min(args.ema_decay, 1 - 1 / (global_step + 1))
                for target, source in zip(ema.parameters(), model.parameters()):
                    target.lerp_(source, 1 - decay)
                for target, source in zip(ema.buffers(), model.buffers()):
                    target.copy_(source)
            count = len(batch["label"])
            seen += count
            loss_sum += float(loss.detach()) * count
            classification_sum += float(classification_loss.detach()) * count
            reconstruction_sum += float(reconstruction_loss.detach()) * count
            retention_sum += float(retention_loss.detach()) * count
            correct += int((logits.argmax(1) == batch["label"]).sum())
            global_step += 1
            if args.limit_train_batches and batch_number + 1 >= args.limit_train_batches:
                break
        values = {
            "epoch": epoch,
            "train/loss": loss_sum / seen,
            "train/classification_loss": classification_sum / seen,
            "train/reconstruction_loss": reconstruction_sum / seen,
            "train/retention_loss": retention_sum / seen,
            "train/accuracy": correct / seen,
            "train/lr": rate,
            "train/epoch_seconds": time.monotonic() - epoch_started,
        }
        if args.fold == 0:
            metrics, arrays = evaluate(ema, valid_loader, device, args.limit_valid_batches)
            values.update(
                {
                    "valid/accuracy": metrics["accuracy"],
                    "valid/correct": metrics["correct"],
                    "valid/nll": metrics["nll"],
                    "valid/visual_missing_accuracy": metrics["visual_missing"]["accuracy"] or 0.0,
                    "valid/worst_user_accuracy": metrics["worst_user_accuracy"],
                }
            )
        print(json.dumps(values), flush=True)
        swanlab.log({key: value for key, value in values.items() if key != "epoch"}, step=epoch)
        with (destination / "history.jsonl").open("a") as handle:
            handle.write(json.dumps(values) + "\n")
        if args.fold == 0:
            current_key = selection_key(metrics)
            if args.select_final and epoch == args.epochs:
                best_key = current_key
                best_epoch = epoch
                best_metrics = metrics
                save_checkpoint(ema, serializable, epoch, metrics, arrays, destination)
            elif not args.select_final and current_key > best_key:
                best_key = current_key
                best_epoch = epoch
                best_metrics = metrics
                save_checkpoint(ema, serializable, epoch, metrics, arrays, destination)

    if args.fold == 1:
        best_metrics, best_arrays = evaluate(ema, valid_loader, device, args.limit_valid_batches)
        best_key = selection_key(best_metrics)
        best_epoch = args.epochs
        save_checkpoint(ema, serializable, best_epoch, best_metrics, best_arrays, destination)
        swanlab.log(
            {
                "blind/accuracy": best_metrics["accuracy"],
                "blind/correct": best_metrics["correct"],
                "blind/nll": best_metrics["nll"],
            },
            step=best_epoch,
        )

    checkpoint = torch.load(destination / "best.pt", map_location="cpu", weights_only=False)
    inference_state = {
        name: value
        for name, value in checkpoint["model"].items()
        if not name.startswith("reconstruction_decoder.")
    }
    quantized, tensor_bytes = quantize_state_dict(inference_state)
    quantized_path = destination / "best_inference_int8.pt"
    torch.save(
        {
            "model_int8": compact_quantized_state_dict(quantized),
            "config": serializable,
            "epoch": checkpoint["epoch"],
            "metrics": checkpoint["metrics"],
            "tensor_bytes": tensor_bytes,
            "excluded_training_only_prefixes": ["reconstruction_decoder."],
        },
        quantized_path,
    )
    yolo = ROOT / "outputs/final_legal_multimodal_v9/models/yolo11n_compact.pt"
    deployment_bytes = quantized_path.stat().st_size + yolo.stat().st_size
    completion = {
        "status": "complete",
        "best_epoch": best_epoch,
        "metrics": best_metrics,
        "selection_key": list(best_key),
        "parameters": serializable["parameters"],
        "inference_parameters": serializable["inference_parameters"],
        "training_only_parameters": serializable["training_only_parameters"],
        "int8_checkpoint_bytes": quantized_path.stat().st_size,
        "deployment_bytes_with_yolo": deployment_bytes,
        "peak_cuda_mb": torch.cuda.max_memory_allocated() / 1e6,
        "elapsed_seconds": time.monotonic() - started,
    }
    try:
        completion["swanlab_url"] = run.url
    except ValueError:
        completion["swanlab_url"] = None
    if deployment_bytes >= 100_000_000:
        raise RuntimeError(f"V80 deployment exceeds 100 MB: {deployment_bytes}")
    (destination / "completion.json").write_text(json.dumps(completion, indent=2) + "\n")
    swanlab.finish()
    print("DONE " + json.dumps(completion), flush=True)


if __name__ == "__main__":
    main()

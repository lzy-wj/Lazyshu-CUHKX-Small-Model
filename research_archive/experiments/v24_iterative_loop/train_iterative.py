#!/usr/bin/env python3
import argparse
import copy
import hashlib
import json
import math
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
sys.path.insert(0, str(HERE))

from iterative_model import build_from_source
from scripts.train_r2plus1d import (
    learning_rate,
    mixed_cross_entropy,
    mixed_stream_cross_entropy,
    mixup_batch,
    prepare_batch,
    seed_everything,
)
from src.data import FOLDS, get_indices, get_records
from src.r2plus1d_data import R2Plus1DDataset
from src.r2plus1d_models import compact_quantized_state_dict, quantize_state_dict


PAIRED_USERS = ({8, 18}, {7, 17})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--iterations", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=HERE / "outputs")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--backbone-lr-factor", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--auxiliary-weight", type=float, default=0.15)
    parser.add_argument("--deep-supervision-decay", type=float, default=0.6)
    parser.add_argument("--feedback-temperature", type=float, default=2.0)
    parser.add_argument("--end-to-end-feedback", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--mixup-alpha", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--limit-train-batches", type=int, default=0)
    parser.add_argument("--limit-valid-batches", type=int, default=0)
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


def subset_metrics(rows, labels, predictions):
    count = int(rows.sum())
    correct = int((labels[rows] == predictions[rows]).sum()) if count else 0
    return {"rows": count, "correct": correct, "accuracy": correct / count if count else None}


@torch.no_grad()
def evaluate(model, loader, device, fold, limit_batches=0):
    model.eval()
    all_probabilities = []
    all_labels = []
    all_indices = []
    all_visual_complete = []
    for batch_number, raw_batch in enumerate(loader):
        batch = prepare_batch(raw_batch, device, multimodal=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(**model_inputs(batch))
        all_probabilities.append(logits.float().softmax(dim=1).cpu())
        all_labels.append(batch["label"].cpu())
        all_indices.append(batch["index"].cpu())
        all_visual_complete.append(batch["visual_mask"].bool().all(dim=1).cpu())
        if limit_batches and batch_number + 1 >= limit_batches:
            break
    probabilities = torch.cat(all_probabilities).numpy()
    labels = torch.cat(all_labels).numpy()
    indices = torch.cat(all_indices).numpy()
    visual_complete = torch.cat(all_visual_complete).numpy().astype(bool)
    predictions = probabilities.argmax(axis=1)
    records = get_records()
    users = np.asarray([records[int(index)]["user"] for index in indices], dtype=np.int64)
    paired = np.isin(users, list(PAIRED_USERS[fold]))
    nll = float(-np.log(probabilities[np.arange(len(labels)), labels].clip(1e-9)).mean())
    report = {
        "rows": len(labels),
        "correct": int((predictions == labels).sum()),
        "accuracy": float((predictions == labels).mean()),
        "nll": nll,
        "visual_complete": subset_metrics(visual_complete, labels, predictions),
        "visual_missing": subset_metrics(~visual_complete, labels, predictions),
        "paired": subset_metrics(paired, labels, predictions),
        "unpaired": subset_metrics(~paired, labels, predictions),
        "per_user": {
            str(user): subset_metrics(users == user, labels, predictions)
            for user in np.unique(users)
        },
    }
    report["worst_user_accuracy"] = min(
        values["accuracy"] for values in report["per_user"].values()
    )
    arrays = {
        "probabilities": probabilities,
        "labels": labels,
        "indices": indices,
        "users": users,
        "visual_complete": visual_complete,
        "paired": paired,
    }
    return report, arrays


def auxiliary_loss(auxiliary, batch, weight, label_smoothing):
    if not auxiliary or weight <= 0:
        return batch["video"].sum() * 0
    losses = []
    for name, logits in auxiliary.items():
        if name == "environment":
            continue
        if name == "video_streams":
            values = mixed_stream_cross_entropy(
                logits, batch, label_smoothing=label_smoothing
            )
            losses.append((values * batch["visual_mask"]).sum() / batch["visual_mask"].sum().clamp_min(1))
        else:
            values = mixed_cross_entropy(
                logits, batch, label_smoothing=label_smoothing, reduction="none"
            )
            mask = (
                torch.ones_like(batch["label"], dtype=torch.float32)
                if name == "video"
                else batch["sensor_mask"][:, {"skeleton": 0, "imu": 1, "radar": 2}[name]]
            )
            losses.append((values * mask).sum() / mask.sum().clamp_min(1))
    return weight * torch.stack(losses).mean() if losses else batch["video"].sum() * 0


def iterative_loss(iteration_logits, batch, decay, label_smoothing):
    steps = iteration_logits.shape[1]
    weights = torch.tensor(
        [decay ** (steps - step - 1) for step in range(steps)],
        device=iteration_logits.device,
        dtype=torch.float32,
    )
    weights /= weights.sum()
    losses = torch.stack(
        [
            mixed_cross_entropy(
                iteration_logits[:, step], batch, label_smoothing=label_smoothing
            )
            for step in range(steps)
        ]
    )
    return (weights * losses).sum(), weights


def selection_key(metrics):
    return (
        int(metrics["correct"]),
        -float(metrics["nll"]),
        int(metrics["visual_missing"]["correct"]),
    )


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA")
    if not 0 < args.deep_supervision_decay <= 1:
        raise ValueError("deep supervision decay must be in (0, 1]")
    seed_everything(args.seed + args.fold)
    device = torch.device("cuda")
    model, source_config, source_checkpoint = build_from_source(
        args.init_checkpoint,
        args.iterations,
        feedback_temperature=args.feedback_temperature,
        detach_feedback=not args.end_to_end_feedback,
    )
    if int(source_config["fold"]) != args.fold:
        raise ValueError("Initialization checkpoint must match the validation fold")
    model.to(device)
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)

    records = get_records()
    train_indices, valid_indices = get_indices(records, args.fold)
    cache_dir = args.cache_dir or Path(source_config["cache_dir"])
    frames = int(source_config.get("frames", 32))
    image_size = int(source_config.get("image_size", 128))
    loader_options = {
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
    generator = torch.Generator().manual_seed(args.seed + args.fold)
    train_loader = DataLoader(
        R2Plus1DDataset(
            train_indices,
            cache_dir,
            training=True,
            include_sensors=True,
            include_radar=bool(source_config.get("use_radar", False)),
            frames=frames,
            image_size=image_size,
            canonical_pose=bool(source_config.get("canonical_pose", False)),
            horizontal_flip_probability=float(source_config.get("horizontal_flip_probability", 0.5)),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
        **loader_options,
    )
    valid_loader = DataLoader(
        R2Plus1DDataset(
            valid_indices,
            cache_dir,
            training=False,
            include_sensors=True,
            include_radar=bool(source_config.get("use_radar", False)),
            frames=frames,
            image_size=image_size,
            canonical_pose=bool(source_config.get("canonical_pose", False)),
        ),
        batch_size=args.batch_size,
        shuffle=False,
        **loader_options,
    )

    backbone_parameters = list(model.base.video.parameters()) + list(model.base.video_head.parameters())
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    head_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in backbone_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "factor": args.backbone_lr_factor},
            {"params": head_parameters, "factor": 1.0},
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    destination = args.output_root / args.name
    destination.mkdir(parents=True, exist_ok=True)
    config = {
        **source_config,
        **vars(args),
        "cache_dir": str(cache_dir),
        "init_checkpoint": str(args.init_checkpoint),
        "output_root": str(args.output_root),
        "fold": args.fold,
        "iterations": args.iterations,
        "detach_feedback": not args.end_to_end_feedback,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "base_parameters": sum(parameter.numel() for parameter in model.base.parameters()),
        "incremental_parameters": model.extra_parameter_count(),
        "train_clips": len(train_indices),
        "valid_clips": len(valid_indices),
        "validation_users": FOLDS[args.fold],
        "precision": "BF16 autocast; FP32 optimizer and EMA",
        "source_checkpoint_sha256": hashlib.sha256(args.init_checkpoint.read_bytes()).hexdigest(),
        "protocol": "Fold0 selects K by correct>NLL>missing-visual correct; Fold1 runs selected K once.",
    }
    serializable = {
        key: str(value) if isinstance(value, Path) else value for key, value in config.items()
    }
    (destination / "config.json").write_text(json.dumps(serializable, indent=2) + "\n")
    (destination / "split.json").write_text(
        json.dumps({"train": train_indices, "valid": valid_indices}) + "\n"
    )

    import swanlab

    run = swanlab.init(
        project="cuhkx-small-multimodal",
        name=args.name,
        config=serializable,
        public=False,
        mode=args.swanlab_mode,
        log_dir=str(HERE / "swanlog"),
        description="V24 shared-weight iterative Transformer refinement ablation.",
    )
    total_steps = len(train_loader) * args.epochs
    warmup_steps = round(len(train_loader) * args.warmup_epochs)
    global_step = 0
    best_key = None
    best_epoch = 0
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        seen = 0
        loss_sum = 0.0
        correct = 0
        epoch_started = time.monotonic()
        for batch_number, raw_batch in enumerate(train_loader):
            rate = learning_rate(global_step, total_steps, warmup_steps, args.lr)
            for group in optimizer.param_groups:
                group["lr"] = rate * group["factor"]
            batch = prepare_batch(raw_batch, device, multimodal=True, training=True)
            mixup_batch(batch, args.mixup_alpha)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, auxiliary, step_logits = model(
                    **model_inputs(batch), return_iteration_logits=True
                )
                task_loss, supervision_weights = iterative_loss(
                    step_logits,
                    batch,
                    args.deep_supervision_decay,
                    args.label_smoothing,
                )
                loss = task_loss + auxiliary_loss(
                    auxiliary, batch, args.auxiliary_weight, args.label_smoothing
                )
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
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
            correct += int((logits.argmax(dim=1) == batch["label"]).sum())
            global_step += 1
            if args.limit_train_batches and batch_number + 1 >= args.limit_train_batches:
                break

        log_values = {
            "train/loss": loss_sum / seen,
            "train/accuracy": correct / seen,
            "train/lr": rate,
            "train/epoch_seconds": time.monotonic() - epoch_started,
        }
        if args.fold == 0:
            metrics, arrays = evaluate(
                ema, valid_loader, device, args.fold, args.limit_valid_batches
            )
            current_key = selection_key(metrics)
            log_values.update({
                "valid/accuracy": metrics["accuracy"],
                "valid/correct": metrics["correct"],
                "valid/nll": metrics["nll"],
                "valid/visual_missing_accuracy": metrics["visual_missing"]["accuracy"] or 0.0,
                "valid/worst_user_accuracy": metrics["worst_user_accuracy"],
            })
        swanlab.log(log_values, step=epoch)
        with (destination / "history.jsonl").open("a") as handle:
            handle.write(json.dumps({"epoch": epoch} | log_values) + "\n")
        print(json.dumps({"epoch": epoch} | log_values), flush=True)
        if args.fold == 0 and (best_key is None or current_key > best_key):
            best_key = current_key
            best_epoch = epoch
            state = {
                "model": {
                    name: value.detach().cpu().half() if value.is_floating_point() else value.detach().cpu()
                    for name, value in ema.state_dict().items()
                },
                "config": serializable,
                "epoch": epoch,
                "metrics": metrics,
            }
            torch.save(state, destination / "best.pt")
            np.savez(destination / "validation.npz", **arrays)
            (destination / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        if args.fold == 0 and epoch - best_epoch >= args.patience:
            break

    if args.fold == 1:
        metrics, arrays = evaluate(
            ema, valid_loader, device, args.fold, args.limit_valid_batches
        )
        best_key = selection_key(metrics)
        best_epoch = epoch
        state = {
            "model": {
                name: value.detach().cpu().half() if value.is_floating_point() else value.detach().cpu()
                for name, value in ema.state_dict().items()
            },
            "config": serializable,
            "epoch": epoch,
            "metrics": metrics,
        }
        torch.save(state, destination / "best.pt")
        np.savez(destination / "validation.npz", **arrays)
        (destination / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        swanlab.log(
            {
                "blind/accuracy": metrics["accuracy"],
                "blind/correct": metrics["correct"],
                "blind/nll": metrics["nll"],
                "blind/visual_missing_accuracy": metrics["visual_missing"]["accuracy"] or 0.0,
                "blind/worst_user_accuracy": metrics["worst_user_accuracy"],
            },
            step=epoch,
        )

    checkpoint = torch.load(destination / "best.pt", map_location="cpu", weights_only=False)
    quantized, tensor_bytes = quantize_state_dict(checkpoint["model"])
    compact = compact_quantized_state_dict(quantized)
    quantized_path = destination / "best_int8.pt"
    torch.save(
        {
            "model_int8": compact,
            "config": serializable,
            "epoch": checkpoint["epoch"],
            "metrics": checkpoint["metrics"],
            "tensor_bytes": tensor_bytes,
        },
        quantized_path,
    )
    completion = {
        "status": "complete",
        "fold": args.fold,
        "iterations": args.iterations,
        "best_epoch": best_epoch,
        "selection_key": list(best_key),
        "metrics": checkpoint["metrics"],
        "parameters": serializable["parameters"],
        "incremental_parameters": serializable["incremental_parameters"],
        "checkpoint_bytes": (destination / "best.pt").stat().st_size,
        "int8_checkpoint_bytes": quantized_path.stat().st_size,
        "int8_tensor_bytes": tensor_bytes,
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

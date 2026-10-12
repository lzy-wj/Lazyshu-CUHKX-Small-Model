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
LOOP_ROOT = ROOT / "experiments/v24_iterative_loop"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LOOP_ROOT))

from train_iterative import (
    auxiliary_loss,
    evaluate,
    iterative_loss,
    learning_rate,
    mixup_batch,
    model_inputs,
    prepare_batch,
    seed_everything,
    selection_key,
)
from src.data import FOLDS, get_indices, get_records
from src.r2plus1d_data import R2Plus1DDataset
from src.r2plus1d_models import compact_quantized_state_dict, quantize_state_dict
from wide_iterative_model import build_from_source


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--iterations", type=int, choices=(1, 2), required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=HERE / "outputs")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--wide-dimension", type=int, default=768)
    parser.add_argument("--wide-depth", type=int, default=4)
    parser.add_argument("--wide-heads", type=int, default=12)
    parser.add_argument("--feedforward-multiplier", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--base-lr-factor", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--auxiliary-weight", type=float, default=0.15)
    parser.add_argument("--deep-supervision-decay", type=float, default=0.6)
    parser.add_argument("--feedback-temperature", type=float, default=2.0)
    parser.add_argument("--end-to-end-feedback", action="store_true")
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--freeze-base", action="store_true")
    parser.add_argument("--retention-weight", type=float, default=0.0)
    parser.add_argument("--retention-temperature", type=float, default=2.0)
    parser.add_argument("--preserve-initial", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--mixup-alpha", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--limit-train-batches", type=int, default=0)
    parser.add_argument("--limit-valid-batches", type=int, default=0)
    parser.add_argument("--swanlab-mode", choices=("cloud", "local", "disabled"), default="cloud")
    return parser.parse_args()


def save_checkpoint(model, config, epoch, metrics, destination, arrays):
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
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA")
    if not 0 < args.deep_supervision_decay <= 1:
        raise ValueError("deep supervision decay must be in (0, 1]")
    if args.residual_scale <= 0 or args.retention_weight < 0 or args.retention_temperature <= 0:
        raise ValueError("Invalid residual or retention setting")
    seed_everything(args.seed + args.fold)
    device = torch.device("cuda")
    model, source_config, _ = build_from_source(
        args.init_checkpoint,
        args.iterations,
        wide_dimension=args.wide_dimension,
        wide_depth=args.wide_depth,
        wide_heads=args.wide_heads,
        feedforward_multiplier=args.feedforward_multiplier,
        feedback_temperature=args.feedback_temperature,
        detach_feedback=not args.end_to_end_feedback,
        residual_scale=args.residual_scale,
    )
    if int(source_config["fold"]) != args.fold:
        raise ValueError("Initialization checkpoint must match the validation fold")
    if args.freeze_base:
        for parameter in model.base.parameters():
            parameter.requires_grad_(False)
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
            horizontal_flip_probability=float(
                source_config.get("horizontal_flip_probability", 0.5)
            ),
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

    base_parameters = [parameter for parameter in model.base.parameters() if parameter.requires_grad]
    base_parameter_ids = {id(parameter) for parameter in base_parameters}
    new_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in base_parameter_ids
    ]
    parameter_groups = [{"params": new_parameters, "factor": 1.0}]
    if base_parameters:
        parameter_groups.insert(0, {"params": base_parameters, "factor": args.base_lr_factor})
    optimizer = torch.optim.AdamW(
        parameter_groups,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    destination = args.output_root / args.name
    destination.mkdir(parents=True, exist_ok=False)
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
        "protocol": "Fold0 compares equal-parameter K1/K2; Fold1 remains blind until selection.",
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
        description="V78 89M-parameter wide shared iterative fusion ablation.",
    )
    total_steps = len(train_loader) * args.epochs
    warmup_steps = round(len(train_loader) * args.warmup_epochs)
    global_step = 0
    best_key = None
    best_epoch = 0
    started = time.monotonic()
    if args.fold == 0 and args.preserve_initial:
        metrics, arrays = evaluate(ema, valid_loader, device, args.fold, args.limit_valid_batches)
        best_key = selection_key(metrics)
        save_checkpoint(ema, serializable, 0, metrics, destination, arrays)
        print(json.dumps({"epoch": 0, "valid/correct": metrics["correct"], "valid/nll": metrics["nll"]}), flush=True)
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
                outputs = model(
                    **model_inputs(batch),
                    return_iteration_logits=True,
                    return_base_logits=args.retention_weight > 0,
                )
                logits, auxiliary, step_logits = outputs[:3]
                task_loss, _ = iterative_loss(
                    step_logits,
                    batch,
                    args.deep_supervision_decay,
                    args.label_smoothing,
                )
                loss = task_loss + auxiliary_loss(
                    auxiliary,
                    batch,
                    args.auxiliary_weight,
                    args.label_smoothing,
                )
                retention_loss = logits.sum() * 0
                if args.retention_weight > 0:
                    base_logits = outputs[3].detach()
                    temperature = args.retention_temperature
                    retention_loss = functional.kl_div(
                        functional.log_softmax(logits.float() / temperature, dim=1),
                        functional.softmax(base_logits.float() / temperature, dim=1),
                        reduction="batchmean",
                    ) * temperature ** 2
                    loss = loss + args.retention_weight * retention_loss
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
        if args.retention_weight > 0:
            log_values["train/retention_loss"] = float(retention_loss.detach())
        if args.fold == 0:
            metrics, arrays = evaluate(
                ema,
                valid_loader,
                device,
                args.fold,
                args.limit_valid_batches,
            )
            current_key = selection_key(metrics)
            log_values.update(
                {
                    "valid/accuracy": metrics["accuracy"],
                    "valid/correct": metrics["correct"],
                    "valid/nll": metrics["nll"],
                    "valid/visual_missing_accuracy": metrics["visual_missing"]["accuracy"] or 0.0,
                    "valid/worst_user_accuracy": metrics["worst_user_accuracy"],
                }
            )
        swanlab.log(log_values, step=epoch)
        with (destination / "history.jsonl").open("a") as handle:
            handle.write(json.dumps({"epoch": epoch} | log_values) + "\n")
        print(json.dumps({"epoch": epoch} | log_values), flush=True)
        if args.fold == 0 and (best_key is None or current_key > best_key):
            best_key = current_key
            best_epoch = epoch
            save_checkpoint(ema, serializable, epoch, metrics, destination, arrays)
        if args.fold == 0 and epoch - best_epoch >= args.patience:
            break

    if args.fold == 1:
        metrics, arrays = evaluate(
            ema,
            valid_loader,
            device,
            args.fold,
            args.limit_valid_batches,
        )
        best_key = selection_key(metrics)
        best_epoch = epoch
        save_checkpoint(ema, serializable, epoch, metrics, destination, arrays)
        swanlab.log(
            {
                "blind/accuracy": metrics["accuracy"],
                "blind/correct": metrics["correct"],
                "blind/nll": metrics["nll"],
            },
            step=epoch,
        )

    checkpoint = torch.load(destination / "best.pt", map_location="cpu", weights_only=False)
    quantized, tensor_bytes = quantize_state_dict(checkpoint["model"])
    quantized_path = destination / "best_int8.pt"
    torch.save(
        {
            "model_int8": compact_quantized_state_dict(quantized),
            "config": serializable,
            "epoch": checkpoint["epoch"],
            "metrics": checkpoint["metrics"],
            "tensor_bytes": tensor_bytes,
        },
        quantized_path,
    )
    yolo = ROOT / "outputs/final_legal_multimodal_v9/models/yolo11n_compact.pt"
    deployment_bytes = quantized_path.stat().st_size + yolo.stat().st_size
    if deployment_bytes >= 100_000_000:
        raise RuntimeError(f"V78 deployment exceeds 100 MB: {deployment_bytes}")
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
        "deployment_bytes_with_yolo": deployment_bytes,
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

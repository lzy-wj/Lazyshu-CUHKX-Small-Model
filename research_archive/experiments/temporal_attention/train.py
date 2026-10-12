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
from torch.utils.data import DataLoader


EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EXPERIMENT_ROOT))

from model import TemporalAttentionR2Plus1DClassifier
from src.data import FOLDS, get_indices, get_records
from src.r2plus1d_data import R2Plus1DDataset
from src.r2plus1d_models import quantize_state_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--fold", type=int, choices=(0,), default=0)
    parser.add_argument("--init-visual", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--backbone-lr-factor", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=float, default=2)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--auxiliary-weight", type=float, default=0.15)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=8)
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
    return maximum * 0.5 * (1 + math.cos(math.pi * min(progress, 1)))


def prepare_batch(batch, device, training=False):
    output = {
        "video": batch["video"].to(device, non_blocking=True).permute(0, 2, 1, 3, 4),
        "label": batch["label"].to(device, non_blocking=True),
        "index": batch["index"],
        "visual_mask": batch["visual_mask"].to(device, non_blocking=True),
        "skeleton": batch["skeleton"].to(device, non_blocking=True),
        "imu": batch["imu"].to(device, non_blocking=True),
        "sensor_mask": batch["sensor_mask"].to(device, non_blocking=True),
    }
    if training:
        drop = torch.rand(output["sensor_mask"].shape, device=device) < 0.1
        output["sensor_mask"] = output["sensor_mask"] * (~drop)
        visual_drop = torch.rand(output["visual_mask"].shape, device=device) < 0.1
        dropped_visual = output["visual_mask"] * (~visual_drop)
        empty = dropped_visual.sum(dim=1) == 0
        dropped_visual[empty] = output["visual_mask"][empty]
        output["visual_mask"] = dropped_visual
    return output


def forward(model, batch):
    return model(
        batch["video"],
        skeleton=batch["skeleton"],
        imu=batch["imu"],
        sensor_mask=batch["sensor_mask"],
        visual_mask=batch["visual_mask"],
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    probabilities = []
    labels = []
    indices = []
    loss_sum = 0.0
    seen = 0
    for raw_batch in loader:
        batch = prepare_batch(raw_batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = forward(model, batch)
            loss = functional.cross_entropy(logits, batch["label"])
        count = len(batch["label"])
        loss_sum += float(loss) * count
        seen += count
        probabilities.append(logits.float().softmax(1).cpu())
        labels.append(batch["label"].cpu())
        indices.append(batch["index"].cpu())
    probabilities = torch.cat(probabilities).numpy()
    labels = torch.cat(labels).numpy()
    indices = torch.cat(indices).numpy()
    metrics = {
        "loss": loss_sum / seen,
        "accuracy": float((probabilities.argmax(1) == labels).mean()),
    }
    return metrics, {"probabilities": probabilities, "labels": labels, "indices": indices}


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA")
    seed_everything(args.seed + args.fold)
    device = torch.device("cuda")
    records = get_records()
    train_indices, valid_indices = get_indices(records, args.fold)
    loader_options = {
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(
        R2Plus1DDataset(
            train_indices,
            args.cache_dir,
            training=True,
            include_sensors=True,
            frames=args.frames,
            image_size=args.image_size,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed + args.fold),
        **loader_options,
    )
    valid_loader = DataLoader(
        R2Plus1DDataset(
            valid_indices,
            args.cache_dir,
            training=False,
            include_sensors=True,
            frames=args.frames,
            image_size=args.image_size,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        **loader_options,
    )
    model = TemporalAttentionR2Plus1DClassifier(
        weights=None,
        dropout=args.dropout,
        input_channels=train_loader.dataset.channels,
    ).to(device)
    source = torch.load(args.init_visual, map_location="cpu", weights_only=False)
    if source["config"]["mode"] != "visual" or int(source["config"]["fold"]) != args.fold:
        raise ValueError("Visual initialization must come from fold 0 visual training")
    current = model.state_dict()
    compatible = {
        name: value
        for name, value in source["model"].items()
        if (name.startswith("video.") or name.startswith("video_head."))
        and name in current and value.shape == current[name].shape
    }
    expected = [name for name in current if name.startswith("video.") or name.startswith("video_head.")]
    if set(compatible) != set(expected):
        raise ValueError("Visual initialization is incomplete")
    model.load_state_dict(compatible, strict=False)
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)

    backbone_parameters = list(model.video.parameters()) + list(model.video_head.parameters())
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    head_parameters = [parameter for parameter in model.parameters() if id(parameter) not in backbone_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "factor": args.backbone_lr_factor},
            {"params": head_parameters, "factor": 1.0},
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    destination = EXPERIMENT_ROOT / "outputs/checkpoints" / args.name
    destination.mkdir(parents=True, exist_ok=True)
    baseline_parameters = 64_955_492
    parameters = sum(parameter.numel() for parameter in model.parameters())
    config = {
        **vars(args),
        "init_visual": str(args.init_visual),
        "cache_dir": str(args.cache_dir),
        "mode": "multimodal",
        "fusion": "relative_time_attention",
        "parameters": parameters,
        "baseline_parameters": baseline_parameters,
        "parameter_delta": parameters - baseline_parameters,
        "input_channels": train_loader.dataset.channels,
        "train_clips": len(train_indices),
        "valid_clips": len(valid_indices),
        "validation_users": FOLDS[args.fold],
        "precision": "BF16 autocast; FP32 optimizer and EMA",
        "architecture": "R(2+1)D-34 CNN + unchanged SkeletonTokens/ConvLSTM + relative-time-aware fusion",
        "scientific_variable": "fusion operator only",
        "visual_initialization_sha256": hashlib.sha256(args.init_visual.read_bytes()).hexdigest(),
        "cache_manifest_sha256": hashlib.sha256((args.cache_dir / "manifest.json").read_bytes()).hexdigest(),
        "source_sha256": {
            name: hashlib.sha256((EXPERIMENT_ROOT / name).read_bytes()).hexdigest()
            for name in ("model.py", "train.py")
        },
    }
    serializable = {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}
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
        log_dir=str(EXPERIMENT_ROOT / "outputs/swanlog"),
        description="Single-variable fold-0 ablation_temporal_attention; no fold-1 or test inspection.",
    )
    print(json.dumps(serializable), flush=True)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = round(len(train_loader) * args.warmup_epochs)
    global_step = 0
    best_accuracy = -1.0
    best_epoch = 0
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_started = time.monotonic()
        loss_sum = 0.0
        correct = 0
        seen = 0
        for raw_batch in train_loader:
            rate = learning_rate(global_step, total_steps, warmup_steps, args.lr)
            for group in optimizer.param_groups:
                group["lr"] = rate * group["factor"]
            batch = prepare_batch(raw_batch, device, training=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, auxiliary = forward(model, batch)
                loss = functional.cross_entropy(
                    logits, batch["label"], label_smoothing=args.label_smoothing
                )
                for name, branch_logits in auxiliary.items():
                    branch_loss = functional.cross_entropy(
                        branch_logits,
                        batch["label"],
                        reduction="none",
                        label_smoothing=args.label_smoothing,
                    )
                    mask = (
                        torch.ones_like(batch["label"], dtype=torch.float32)
                        if name == "video"
                        else batch["sensor_mask"][:, 0 if name == "skeleton" else 1]
                    )
                    loss = loss + args.auxiliary_weight / len(auxiliary) * (
                        branch_loss * mask
                    ).sum() / mask.sum().clamp_min(1)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            with torch.no_grad():
                decay = min(args.ema_decay, 1 - 1 / (global_step + 1))
                for target, source_parameter in zip(ema.parameters(), model.parameters()):
                    target.lerp_(source_parameter, 1 - decay)
                for target, source_buffer in zip(ema.buffers(), model.buffers()):
                    target.copy_(source_buffer)
            count = len(batch["label"])
            loss_sum += float(loss.detach()) * count
            correct += int((logits.argmax(1) == batch["label"]).sum())
            seen += count
            global_step += 1
        validation, predictions = evaluate(ema, valid_loader, device)
        metrics = {
            "train/loss": loss_sum / seen,
            "train/accuracy": correct / seen,
            "train/lr": rate,
            "valid/loss": validation["loss"],
            "valid/accuracy": validation["accuracy"],
            "system/epoch_seconds": time.monotonic() - epoch_started,
            "system/peak_cuda_mb": torch.cuda.max_memory_allocated() / 1e6,
        }
        if validation["accuracy"] > best_accuracy:
            best_accuracy = validation["accuracy"]
            best_epoch = epoch
            state = {
                "model": {
                    name: value.detach().cpu().half() if value.is_floating_point() else value.detach().cpu()
                    for name, value in ema.state_dict().items()
                },
                "config": serializable,
                "epoch": epoch,
                "metrics": validation,
            }
            torch.save(state, destination / "best.pt")
            np.savez(destination / "validation.npz", **predictions)
            (destination / "metrics.json").write_text(
                json.dumps(validation | {"epoch": epoch}, indent=2) + "\n"
            )
        metrics["valid/best_accuracy"] = best_accuracy
        with (destination / "history.jsonl").open("a") as handle:
            handle.write(json.dumps({"epoch": epoch} | metrics) + "\n")
        swanlab.log(metrics, step=epoch)
        print(
            f"epoch={epoch} " + " ".join(f"{name}={value:.5f}" for name, value in metrics.items()),
            flush=True,
        )
        if epoch - best_epoch >= args.patience:
            print("Early stopping", flush=True)
            break

    checkpoint = torch.load(destination / "best.pt", map_location="cpu", weights_only=False)
    quantized, tensor_bytes = quantize_state_dict(checkpoint["model"])
    quantized_path = destination / "best_int8.pt"
    torch.save(
        {
            "model_int8": quantized,
            "config": serializable,
            "epoch": checkpoint["epoch"],
            "metrics": checkpoint["metrics"],
            "tensor_bytes": tensor_bytes,
        },
        quantized_path,
    )
    try:
        swanlab_url = run.url
    except ValueError:
        swanlab_url = None
    completion = {
        "status": "complete",
        "best_accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "checkpoint_bytes": (destination / "best.pt").stat().st_size,
        "quantized_checkpoint_bytes": quantized_path.stat().st_size,
        "quantized_tensor_bytes": tensor_bytes,
        "parameters": parameters,
        "baseline_parameters": baseline_parameters,
        "parameter_delta": parameters - baseline_parameters,
        "peak_cuda_mb": torch.cuda.max_memory_allocated() / 1e6,
        "elapsed_seconds": time.monotonic() - started,
        "swanlab_url": swanlab_url,
    }
    (destination / "completion.json").write_text(json.dumps(completion, indent=2) + "\n")
    swanlab.log({"system/checkpoint_bytes": completion["checkpoint_bytes"]})
    swanlab.finish()
    print("DONE " + json.dumps(completion), flush=True)


if __name__ == "__main__":
    main()

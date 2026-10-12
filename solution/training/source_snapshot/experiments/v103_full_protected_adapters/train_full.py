#!/usr/bin/env python3
import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.v96_modality_specialized_backbone.train import learning_rate, update_ema
from experiments.v97_protected_visual_adapters.model import build_from_source
from scripts.train_r2plus1d import (
    mixed_cross_entropy,
    mixed_stream_cross_entropy,
    mixup_batch,
    prepare_batch,
    seed_everything,
)
from src.data import get_records
from src.r2plus1d_data import R2Plus1DDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=HERE / "outputs")
    parser.add_argument("--location", choices=("intermediate", "dual"), required=True)
    parser.add_argument("--bottleneck", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--mixup-alpha", type=float, default=0.3)
    parser.add_argument("--auxiliary-weight", type=float, default=0.15)
    parser.add_argument("--retention-weight", type=float, default=0.5)
    parser.add_argument("--retention-temperature", type=float, default=2.0)
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


def adapter_state(model):
    prefixes = (
        "base.video.intermediate_adapters.",
        "base.video.output_adapters.",
    )
    return {
        name: value.detach().cpu().half() if value.is_floating_point() else value.detach().cpu()
        for name, value in model.state_dict().items()
        if name.startswith(prefixes)
    }


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda")
    model, teacher, source_config = build_from_source(
        args.init_checkpoint,
        bottleneck=args.bottleneck,
        output_adapters=args.location == "dual",
    )
    if int(source_config["fold"]) != -1:
        raise ValueError("V103 requires a full-data source checkpoint")
    model.to(device)
    teacher.to(device)
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    trainable_names = {name for name, _ in trainable}

    records = get_records()
    train_indices = [record["index"] for record in records if record["split"] == "train"]
    train_loader = DataLoader(
        R2Plus1DDataset(
            train_indices,
            args.cache_dir,
            training=True,
            include_sensors=True,
            include_radar=False,
            frames=32,
            image_size=160,
            horizontal_flip_probability=float(source_config.get("horizontal_flip_probability", 0.5)),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
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
        "train_clips": len(train_indices),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for _, parameter in trainable),
        "source_checkpoint_sha256": hashlib.sha256(args.init_checkpoint.read_bytes()).hexdigest(),
        "selection": "fixed six-epoch full-data refit from dual-fold protocol",
        "precision": "BF16 autocast; FP32 optimizer and EMA",
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
        description="Full-data protected visual adapter refit at native 160 pixels.",
    )
    started = time.monotonic()
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = {"loss": 0.0, "classification": 0.0, "auxiliary": 0.0, "retention": 0.0}
        seen = 0
        correct = 0
        epoch_started = time.monotonic()
        for raw_batch in train_loader:
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
                classification = mixed_cross_entropy(logits, batch, label_smoothing=args.label_smoothing)
                stream_losses = mixed_stream_cross_entropy(
                    details["video_streams"], batch, label_smoothing=args.label_smoothing
                )
                visual_mask = batch["visual_mask"].float()
                auxiliary = (stream_losses * visual_mask).sum() / visual_mask.sum().clamp_min(1)
                temperature = args.retention_temperature
                retention = functional.kl_div(
                    functional.log_softmax(logits.float() / temperature, dim=1),
                    functional.softmax(teacher_logits.float() / temperature, dim=1),
                    reduction="batchmean",
                ) * temperature ** 2
                loss = classification + args.auxiliary_weight * auxiliary + args.retention_weight * retention
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite V103 loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([parameter for _, parameter in trainable], 5)
            optimizer.step()
            update_ema(
                ema,
                model,
                trainable_names,
                min(args.ema_decay, 1 - 1 / (global_step + 1)),
            )
            count = len(batch["label"])
            seen += count
            correct += int((logits.argmax(dim=1) == batch["label"]).sum())
            sums["loss"] += float(loss.detach()) * count
            sums["classification"] += float(classification.detach()) * count
            sums["auxiliary"] += float(auxiliary.detach()) * count
            sums["retention"] += float(retention.detach()) * count
            global_step += 1
        values = {
            "epoch": epoch,
            "train/loss": sums["loss"] / seen,
            "train/classification": sums["classification"] / seen,
            "train/auxiliary": sums["auxiliary"] / seen,
            "train/retention": sums["retention"] / seen,
            "train/accuracy": correct / seen,
            "train/lr": rate,
            "system/epoch_seconds": time.monotonic() - epoch_started,
            "system/peak_cuda_mb": torch.cuda.max_memory_allocated() / 1e6,
        }
        with (destination / "history.jsonl").open("a") as handle:
            handle.write(json.dumps(values) + "\n")
        print(json.dumps(values), flush=True)
        swanlab.log({key: value for key, value in values.items() if key != "epoch"}, step=epoch)

    state = adapter_state(ema)
    torch.save({"adapter": state, "config": serializable}, destination / "adapter.pt")
    completion = {
        "status": "complete",
        "train_clips": len(train_indices),
        "parameters": serializable["parameters"],
        "trainable_parameters": serializable["trainable_parameters"],
        "adapter_bytes": (destination / "adapter.pt").stat().st_size,
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

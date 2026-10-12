#!/usr/bin/env python3
import argparse
import copy
import hashlib
import json
import random
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

from experiments.v97_protected_visual_adapters.model import build_from_source
from experiments.v98_cross_modal_target_adapter.adapt import stream_distillation_loss
from scripts.evaluate_r2plus1d_checkpoint import (
    adapt_batch_norm,
    make_loader,
    model_inputs,
    predict,
)
from src.data import get_records
from src.r2plus1d_data import R2Plus1DDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pseudo-labels", type=Path, required=True)
    parser.add_argument("--soft-targets", type=Path)
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--source-user", type=int, required=True)
    parser.add_argument("--additional-source-users", type=int, nargs="*", default=[])
    parser.add_argument("--target-user", type=int, required=True)
    parser.add_argument(
        "--adapter-mode", choices=("visual", "multimodal", "heads"), default="visual"
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=HERE / "outputs")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--confidence-threshold", type=float, default=0.6)
    parser.add_argument("--source-supervision-weight", type=float, default=1.0)
    parser.add_argument("--target-confidence-floor", type=float, default=1.0)
    parser.add_argument("--retention-weight", type=float, default=0.25)
    parser.add_argument("--stream-distillation-weight", type=float, default=0.15)
    parser.add_argument("--target-soft-weight", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--tta-crop-scale", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--swanlab-mode", choices=("cloud", "local", "disabled"), default="cloud")
    return parser.parse_args()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    args = parse_args()
    if args.source_supervision_weight <= 0:
        raise ValueError("Source supervision weight must be positive")
    if not 0 <= args.target_confidence_floor <= 1:
        raise ValueError("Target confidence floor must be in [0, 1]")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    builder = build_from_source
    if args.adapter_mode == "multimodal":
        from experiments.v118_multimodal_paired_adapters.model import (
            build_from_source as multimodal_builder,
        )

        builder = multimodal_builder
    elif args.adapter_mode == "heads":
        from experiments.v123_paired_head_adapters.model import (
            build_from_source as head_builder,
        )

        builder = head_builder
    model, teacher, config = builder(
        args.checkpoint, bottleneck=128, output_adapters=True
    )
    if int(config["fold"]) != args.fold:
        raise ValueError("Checkpoint and fold disagree")
    model.cuda().eval()
    teacher.cuda().eval()
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    initial = {
        name: parameter.detach().cpu().clone() for name, parameter in trainable
    }

    records = get_records()
    source_users = {args.source_user, *args.additional_source_users}
    source_indices = [
        record["index"]
        for record in records
        if record["split"] == "train" and record["user"] in source_users
    ]
    target_indices = [
        record["index"]
        for record in records
        if record["split"] == "train" and record["user"] == args.target_user
    ]
    if not source_indices or not target_indices:
        raise ValueError("Source or target user has no rows")
    source_set = set(source_indices)
    target_set = set(target_indices)
    archive = np.load(args.pseudo_labels)
    positions = {int(index): position for position, index in enumerate(archive["indices"])}
    supervision = torch.full(
        (len(records),), -1, dtype=torch.long, device="cuda"
    )
    active_rows = torch.zeros(len(records), dtype=torch.bool, device="cuda")
    source_rows = torch.zeros(len(records), dtype=torch.bool, device="cuda")
    confidence = torch.zeros(len(records), dtype=torch.float32, device="cuda")
    soft_targets = None
    if args.target_soft_weight < 0:
        raise ValueError("Target soft weight must be nonnegative")
    if args.target_soft_weight and args.soft_targets is None:
        raise ValueError("A positive target soft weight requires --soft-targets")
    if args.soft_targets is not None:
        soft_archive = np.load(args.soft_targets)
        soft_positions = {
            int(index): position for position, index in enumerate(soft_archive["indices"])
        }
        soft_targets = torch.zeros(
            (len(records), 40), dtype=torch.float32, device="cuda"
        )
    for index in source_indices:
        supervision[index] = int(records[index]["label"])
        active_rows[index] = True
        source_rows[index] = True
        confidence[index] = 1.0
    selected_target_rows = 0
    for index in target_indices:
        position = positions[index]
        supervision[index] = int(archive["pseudo_labels"][position])
        use_target = bool(
            archive["paired"][position]
            or archive["confidence"][position] >= args.confidence_threshold
        )
        active_rows[index] = use_target
        confidence[index] = float(archive["confidence"][position])
        if soft_targets is not None:
            soft_targets[index] = torch.as_tensor(
                soft_archive["probabilities"][soft_positions[index]],
                dtype=torch.float32,
                device="cuda",
            )
        selected_target_rows += int(use_target)

    training_indices = source_indices + target_indices
    loader = DataLoader(
        R2Plus1DDataset(
            training_indices,
            args.cache_dir,
            training=True,
            include_sensors=True,
            include_radar=False,
            frames=args.frames,
            image_size=args.image_size,
            photometric_jitter=0.05,
            horizontal_flip_probability=0.5,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable], lr=args.lr, weight_decay=1e-4
    )
    destination = args.output_root / args.name
    destination.mkdir(parents=True, exist_ok=False)
    run_config = {
        **vars(args),
        "checkpoint": str(args.checkpoint),
        "pseudo_labels": str(args.pseudo_labels),
        "cache_dir": str(args.cache_dir),
        "output_root": str(args.output_root),
        "source_rows": len(source_indices),
        "source_users": sorted(source_users),
        "target_rows": len(target_indices),
        "selected_target_rows": selected_target_rows,
        "trainable_parameters": sum(parameter.numel() for _, parameter in trainable),
        "checkpoint_sha256": sha256(args.checkpoint),
        "pseudo_labels_sha256": sha256(args.pseudo_labels),
        "soft_targets_sha256": sha256(args.soft_targets) if args.soft_targets else None,
        "architecture": (
            "frozen multimodal CNN/RNN/Transformer with paired-person "
            f"{args.adapter_mode} spatiotemporal CNN adapters"
        ),
    }
    serializable = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in run_config.items()
    }
    (destination / "config.json").write_text(json.dumps(serializable, indent=2) + "\n")

    import swanlab

    run = swanlab.init(
        project="cuhkx-small-multimodal",
        name=args.name,
        config=serializable,
        public=False,
        mode=args.swanlab_mode,
        log_dir=str(HERE / "swanlog"),
        description="Paired-person true-source plus pseudo-target visual adaptation.",
    )
    started = time.monotonic()
    history = []
    for epoch in range(1, args.epochs + 1):
        sums = {
            "loss": 0.0,
            "supervision": 0.0,
            "retention": 0.0,
            "stream": 0.0,
            "target_soft": 0.0,
        }
        rows = 0
        true_rows = 0
        pseudo_rows = 0
        for raw_batch in loader:
            indices = raw_batch["index"].cuda(non_blocking=True)
            active = active_rows[indices]
            if not active.any():
                continue
            video, keyword = model_inputs(raw_batch, "shared_multimodal")
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                teacher_logits, _ = teacher(video, **keyword)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, details = model(video, **keyword)
                row_loss = functional.cross_entropy(
                    logits[active],
                    supervision[indices][active],
                    label_smoothing=0.03,
                    reduction="none",
                )
                batch_source = source_rows[indices][active]
                row_weights = args.target_confidence_floor + (
                    1.0 - args.target_confidence_floor
                ) * confidence[indices][active]
                row_weights = torch.where(
                    batch_source,
                    torch.full_like(row_weights, args.source_supervision_weight),
                    row_weights,
                )
                classification = (row_loss * row_weights).sum() / row_weights.sum()
                retention = functional.kl_div(
                    functional.log_softmax(logits.float(), dim=1),
                    functional.softmax(teacher_logits.float(), dim=1),
                    reduction="batchmean",
                )
                stream = (
                    stream_distillation_loss(
                        details["video_streams"],
                        teacher_logits,
                        keyword["visual_mask"],
                        2.0,
                    )
                    if "video_streams" in details
                    else logits.new_zeros(())
                )
                target_active = active & ~source_rows[indices]
                target_soft = logits.new_zeros(())
                if soft_targets is not None and target_active.any():
                    temperature = 2.0
                    target_soft = functional.kl_div(
                        functional.log_softmax(
                            logits[target_active].float() / temperature, dim=1
                        ),
                        soft_targets[indices][target_active],
                        reduction="batchmean",
                    ) * temperature ** 2
                loss = (
                    classification
                    + args.retention_weight * retention
                    + args.stream_distillation_weight * stream
                    + args.target_soft_weight * target_soft
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for _, parameter in trainable], 2.0
            )
            optimizer.step()
            count = int(active.sum())
            current_source = int((source_rows[indices] & active).sum())
            rows += count
            true_rows += current_source
            pseudo_rows += count - current_source
            sums["loss"] += float(loss.detach()) * count
            sums["supervision"] += float(classification.detach()) * count
            sums["retention"] += float(retention.detach()) * count
            sums["stream"] += float(stream.detach()) * count
            sums["target_soft"] += float(target_soft.detach()) * count
        values = {
            "epoch": epoch,
            "train/loss": sums["loss"] / rows,
            "train/supervision": sums["supervision"] / rows,
            "train/retention": sums["retention"] / rows,
            "train/stream": sums["stream"] / rows,
            "train/target_soft": sums["target_soft"] / rows,
            "train/true_source_rows": true_rows,
            "train/pseudo_target_rows": pseudo_rows,
        }
        history.append(values)
        swanlab.log({key: value for key, value in values.items() if key != "epoch"}, step=epoch)
        print(json.dumps(values), flush=True)

    loader_args = copy.copy(args)
    loader_args.tta_hflip = True
    target_loader = make_loader(target_indices, loader_args, config, "shared_multimodal")
    adapt_batch_norm(model, target_loader, "shared_multimodal", passes=1, reset=True)
    predicted = predict(
        model,
        target_loader,
        "shared_multimodal",
        True,
        args.tta_crop_scale,
    )
    ordered = {index: (label, values) for index, label, values, _ in predicted}
    indices = np.asarray(target_indices, dtype=np.int64)
    labels = np.asarray([ordered[int(index)][0] for index in indices])
    probabilities = np.stack([ordered[int(index)][1] for index in indices])
    np.savez(
        destination / "validation.npz",
        probabilities=probabilities,
        labels=labels,
        indices=indices,
    )
    delta = {
        name: (parameter.detach().cpu() - initial[name]).half()
        for name, parameter in trainable
    }
    torch.save({"delta": delta, "config": serializable}, destination / "adapter.pt")
    report = {
        "rows": len(labels),
        "correct": int((probabilities.argmax(1) == labels).sum()),
        "accuracy": float((probabilities.argmax(1) == labels).mean()),
        "adapter_bytes": (destination / "adapter.pt").stat().st_size,
        "elapsed_seconds": time.monotonic() - started,
        "history": history,
    }
    try:
        report["swanlab_url"] = run.url
    except ValueError:
        report["swanlab_url"] = None
    (destination / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    swanlab.finish()
    print("DONE " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

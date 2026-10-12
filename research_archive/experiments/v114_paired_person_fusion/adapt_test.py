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
from torch.nn import functional
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.v104_domain_gated_adapters.adapt_test import retained_state
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
    parser.add_argument("--source-user", type=int, required=True)
    parser.add_argument("--target-dates", nargs="+", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=HERE / "outputs")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--retention-weight", type=float, default=0.25)
    parser.add_argument("--stream-distillation-weight", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--swanlab-mode", choices=("cloud", "local", "disabled"), default="cloud"
    )
    return parser.parse_args()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model, teacher, source_config = build_from_source(
        args.checkpoint, bottleneck=128, output_adapters=True
    )
    if int(source_config["fold"]) != -1:
        raise ValueError("Full adaptation requires a full-data checkpoint")
    model.cuda().eval()
    teacher.cuda().eval()
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]

    records = get_records()
    source_indices = [
        record["index"]
        for record in records
        if record["split"] == "train" and record["user"] == args.source_user
    ]
    archive = np.load(args.pseudo_labels)
    archive_positions = {
        int(index): position for position, index in enumerate(archive["indices"])
    }
    target_dates = set(args.target_dates)
    target_indices = [
        int(index)
        for index, date in zip(archive["indices"], archive["dates"])
        if str(date) in target_dates
    ]
    if not source_indices or not target_indices:
        raise ValueError("Source or target selection is empty")
    supervision = torch.full(
        (len(records),), -1, dtype=torch.long, device="cuda"
    )
    source_rows = torch.zeros(len(records), dtype=torch.bool, device="cuda")
    for index in source_indices:
        supervision[index] = int(records[index]["label"])
        source_rows[index] = True
    for index in target_indices:
        supervision[index] = int(
            archive["pseudo_labels"][archive_positions[index]]
        )

    training_indices = source_indices + target_indices
    loader = DataLoader(
        R2Plus1DDataset(
            training_indices,
            args.cache_dir,
            training=True,
            include_sensors=True,
            include_radar=False,
            frames=32,
            image_size=160,
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
        [parameter for _, parameter in trainable],
        lr=args.lr,
        weight_decay=1e-4,
    )
    destination = args.output_root / args.name
    destination.mkdir(parents=True, exist_ok=False)
    config = {
        **vars(args),
        "checkpoint": str(args.checkpoint),
        "pseudo_labels": str(args.pseudo_labels),
        "cache_dir": str(args.cache_dir),
        "output_root": str(args.output_root),
        "source_rows": len(source_indices),
        "target_rows": len(target_indices),
        "trainable_parameters": sum(
            parameter.numel() for _, parameter in trainable
        ),
        "checkpoint_sha256": sha256(args.checkpoint),
        "pseudo_labels_sha256": sha256(args.pseudo_labels),
        "architecture": (
            "frozen multimodal CNN/RNN/Transformer with paired-person "
            "spatiotemporal CNN adapters"
        ),
        "precision": "BF16 autocast; FP32 optimizer",
    }
    serializable = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in config.items()
    }
    (destination / "config.json").write_text(
        json.dumps(serializable, indent=2) + "\n"
    )

    import swanlab

    run = swanlab.init(
        project="cuhkx-small-multimodal",
        name=args.name,
        config=serializable,
        public=False,
        mode=args.swanlab_mode,
        log_dir=str(HERE / "swanlog"),
        description="Full-data paired-person visual adapter.",
    )
    started = time.monotonic()
    history = []
    for epoch in range(1, args.epochs + 1):
        sums = {"loss": 0.0, "supervision": 0.0, "retention": 0.0, "stream": 0.0}
        rows = 0
        true_rows = 0
        for raw_batch in loader:
            indices = raw_batch["index"].cuda(non_blocking=True)
            video, keyword = model_inputs(raw_batch, "shared_multimodal")
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                teacher_logits, _ = teacher(video, **keyword)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, details = model(video, **keyword)
                classification = functional.cross_entropy(
                    logits, supervision[indices], label_smoothing=0.03
                )
                retention = functional.kl_div(
                    functional.log_softmax(logits.float(), dim=1),
                    functional.softmax(teacher_logits.float(), dim=1),
                    reduction="batchmean",
                )
                stream = stream_distillation_loss(
                    details["video_streams"],
                    teacher_logits,
                    keyword["visual_mask"],
                    2.0,
                )
                loss = (
                    classification
                    + args.retention_weight * retention
                    + args.stream_distillation_weight * stream
                )
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite V114 loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for _, parameter in trainable], 2.0
            )
            optimizer.step()
            count = len(indices)
            rows += count
            true_rows += int(source_rows[indices].sum())
            sums["loss"] += float(loss.detach()) * count
            sums["supervision"] += float(classification.detach()) * count
            sums["retention"] += float(retention.detach()) * count
            sums["stream"] += float(stream.detach()) * count
        values = {
            "epoch": epoch,
            "train/loss": sums["loss"] / rows,
            "train/supervision": sums["supervision"] / rows,
            "train/retention": sums["retention"] / rows,
            "train/stream": sums["stream"] / rows,
            "train/true_source_rows": true_rows,
            "train/pseudo_target_rows": rows - true_rows,
        }
        history.append(values)
        swanlab.log(
            {key: value for key, value in values.items() if key != "epoch"},
            step=epoch,
        )
        print(json.dumps(values), flush=True)

    loader_args = argparse.Namespace(
        cache_dir=args.cache_dir,
        frames=32,
        image_size=160,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    target_loader = make_loader(
        target_indices, loader_args, source_config, "shared_multimodal"
    )
    adapt_batch_norm(model, target_loader, "shared_multimodal", passes=1, reset=True)
    predicted = predict(model, target_loader, "shared_multimodal", True, 0.75)
    by_index = {index: probabilities for index, _, probabilities, _ in predicted}
    probabilities = np.stack([by_index[index] for index in target_indices])
    np.savez(
        destination / "predictions.npz",
        probabilities=probabilities,
        indices=np.asarray(target_indices, dtype=np.int64),
    )
    torch.save(
        {"state": retained_state(model), "config": serializable},
        destination / "adapted_state.pt",
    )
    report = {
        "status": "complete",
        "source_rows": len(source_indices),
        "target_rows": len(target_indices),
        "adapted_state_bytes": (destination / "adapted_state.pt").stat().st_size,
        "mean_confidence": float(probabilities.max(axis=1).mean()),
        "peak_cuda_mb": torch.cuda.max_memory_allocated() / 1e6,
        "elapsed_seconds": time.monotonic() - started,
        "history": history,
    }
    try:
        report["swanlab_url"] = run.url
    except ValueError:
        report["swanlab_url"] = None
    (destination / "completion.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    swanlab.finish()
    print("DONE " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

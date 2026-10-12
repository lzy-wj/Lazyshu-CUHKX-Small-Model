#!/usr/bin/env python3
import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.adapt_paired_source import checkpoint_state, model_inputs, select_trainable
from scripts.evaluate_r2plus1d_checkpoint import adapt_batch_norm, build_model, make_loader, predict
from src.data import get_indices, get_records
from src.r2plus1d_data import R2Plus1DDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-targets", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1))
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--scope", choices=("heads", "fusion_last"), default="heads")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--confidence-threshold", type=float, default=0.6)
    parser.add_argument("--structure-weight", type=float, default=0.20)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--retention-weight", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--tta-hflip", action="store_true")
    parser.add_argument("--tta-crop-scale", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter-output", type=Path)
    return parser.parse_args()


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float32)
    probabilities = np.clip(probabilities, 1e-8, None)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def blended_targets(raw_probabilities, structured_probabilities, structure_weight, smoothing):
    if not 0 <= structure_weight <= 1:
        raise ValueError("structure weight must be in [0, 1]")
    if not 0 <= smoothing < 1:
        raise ValueError("label smoothing must be in [0, 1)")
    raw = normalize(raw_probabilities)
    structured_labels = normalize(structured_probabilities).argmax(axis=1)
    classes = raw.shape[1]
    structured = np.full_like(raw, smoothing / classes)
    structured[np.arange(len(raw)), structured_labels] += 1 - smoothing
    targets = (1 - structure_weight) * raw + structure_weight * structured
    return normalize(targets)


def main():
    args = parse_args()
    if args.test == (args.fold is not None):
        raise ValueError("Select exactly one of --test or --fold")
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    mode = config["mode"]
    state = checkpoint_state(checkpoint)
    model = build_model(config, copy.deepcopy(state))
    retention_teacher = build_model(config, copy.deepcopy(state))
    selected_parameters = select_trainable(model, args.scope)
    initial = {name: parameter.detach().cpu().clone() for name, parameter in selected_parameters}

    records = get_records()
    if args.test:
        target_indices = [record["index"] for record in records if record["split"] == "test"]
    else:
        _, target_indices = get_indices(records, args.fold)
    archive = np.load(args.teacher_targets)
    if not np.array_equal(archive["indices"], np.asarray(target_indices)):
        raise ValueError("Teacher targets are not aligned with the requested split")
    raw_probabilities = archive["raw_probabilities"]
    targets = blended_targets(
        raw_probabilities,
        archive["probabilities"],
        args.structure_weight,
        args.label_smoothing,
    )
    confidence = archive["confidence"] if "confidence" in archive else normalize(raw_probabilities).max(1)
    paired = archive["paired"].astype(bool)
    active_rows = paired | (confidence >= args.confidence_threshold)
    positions = {int(index): position for position, index in enumerate(target_indices)}

    dataset = R2Plus1DDataset(
        target_indices,
        args.cache_dir,
        training=True,
        include_sensors=mode in ("multimodal", "shared_multimodal"),
        include_radar=bool(config.get("use_radar", False)),
        canonical_pose=bool(config.get("canonical_pose", False)),
        frames=args.frames or int(config.get("frames", 16)),
        image_size=int(config.get("image_size", 128)),
        photometric_jitter=0.05,
        horizontal_flip_probability=0.5,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in selected_parameters],
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history = []
    model.eval()
    retention_teacher.eval()
    for epoch in range(args.epochs):
        loss_sum = 0.0
        active_count = 0
        for batch in loader:
            batch_indices = batch["index"].numpy()
            batch_positions = np.asarray([positions[int(index)] for index in batch_indices])
            active = torch.from_numpy(active_rows[batch_positions]).cuda(non_blocking=True)
            if not active.any():
                continue
            target = torch.from_numpy(targets[batch_positions]).cuda(non_blocking=True)
            video, keyword = model_inputs(batch, mode)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                retained_logits, _ = retention_teacher(video, **keyword)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(video, **keyword)
                soft_cross_entropy = -(
                    target[active] * functional.log_softmax(logits[active].float(), dim=1)
                ).sum(dim=1).mean()
                retention = functional.kl_div(
                    functional.log_softmax(logits.float(), dim=1),
                    functional.softmax(retained_logits.float(), dim=1),
                    reduction="batchmean",
                )
                loss = soft_cross_entropy + args.retention_weight * retention
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for _, parameter in selected_parameters],
                2.0,
            )
            optimizer.step()
            rows = int(active.sum())
            loss_sum += float(loss.detach()) * rows
            active_count += rows
        scheduler.step()
        history.append(
            {
                "epoch": epoch + 1,
                "loss": loss_sum / max(active_count, 1),
                "active_rows": active_count,
                "lr": scheduler.get_last_lr()[0],
            }
        )
        print(json.dumps(history[-1]))

    del retention_teacher
    target_loader = make_loader(target_indices, args, config, mode)
    adapt_batch_norm(model, target_loader, mode, passes=1, reset=True)
    predicted = predict(model, target_loader, mode, args.tta_hflip, args.tta_crop_scale)
    ordered = {index: (label, values) for index, label, values, _ in predicted}
    indices = np.asarray(target_indices, dtype=np.int64)
    labels = np.asarray([ordered[int(index)][0] for index in indices])
    probabilities = np.stack([ordered[int(index)][1] for index in indices])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = {"probabilities": probabilities, "indices": indices}
    if not args.test:
        output["labels"] = labels
    np.savez(args.output, **output)
    adapter_path = args.adapter_output or args.output.with_suffix(".adapter.pt")
    delta = {
        name: (parameter.detach().cpu() - initial[name]).half()
        for name, parameter in selected_parameters
    }
    torch.save(
        {
            "delta": delta,
            "scope": args.scope,
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "teacher_targets_sha256": file_sha256(args.teacher_targets),
            "structure_weight": args.structure_weight,
            "label_smoothing": args.label_smoothing,
            "confidence_threshold": args.confidence_threshold,
            "epochs": args.epochs,
            "lr": args.lr,
            "retention_weight": args.retention_weight,
        },
        adapter_path,
    )
    report = {
        "checkpoint": str(args.checkpoint),
        "split": "test" if args.test else f"fold{args.fold}",
        "rows": len(indices),
        "selected_teacher_rows": int(active_rows.sum()),
        "scope": args.scope,
        "trainable_parameters": sum(parameter.numel() for _, parameter in selected_parameters),
        "adapter": str(adapter_path),
        "adapter_bytes": adapter_path.stat().st_size,
        "structure_weight": args.structure_weight,
        "label_smoothing": args.label_smoothing,
        "confidence_threshold": args.confidence_threshold,
        "mean_confidence": float(probabilities.max(axis=1).mean()),
        "history": history,
    }
    if not args.test:
        report["accuracy"] = float((probabilities.argmax(axis=1) == labels).mean())
        report["correct"] = int((probabilities.argmax(axis=1) == labels).sum())
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

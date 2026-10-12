#!/usr/bin/env python3
import argparse
import copy
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

from scripts.evaluate_r2plus1d_checkpoint import adapt_batch_norm, build_model, make_loader, predict
from src.data import get_records
from src.r2plus1d_data import R2Plus1DDataset
from src.r2plus1d_models import dequantize_state_dict


HEAD_PREFIXES = (
    "video_head.",
    "head.",
    "skeleton_head.",
    "imu_head.",
    "radar_head.",
    "normalization.",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-user", type=int, required=True)
    parser.add_argument("--target-user", type=int, required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--scope", choices=("heads", "fusion_last"), default="heads")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--distillation-weight", type=float, default=0.25)
    parser.add_argument("--auxiliary-weight", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--adabn", choices=("none", "reset", "keep"), default="reset")
    parser.add_argument("--tta-hflip", action="store_true")
    parser.add_argument("--tta-crop-scale", type=float, default=1.0)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter-output", type=Path)
    return parser.parse_args()


def checkpoint_state(checkpoint):
    for key in ("model_int4", "model_int7", "model_int6", "model_int8"):
        if key in checkpoint:
            return dequantize_state_dict(checkpoint[key])
    return checkpoint["model"]


def select_trainable(model, scope):
    last_fusion = len(model.fusion.layers) - 1
    selected = []
    for name, parameter in model.named_parameters():
        trainable = name.startswith(HEAD_PREFIXES)
        if scope == "fusion_last":
            trainable |= name.startswith(f"fusion.layers.{last_fusion}.")
            trainable |= name in ("class_token", "modality_embedding")
        parameter.requires_grad_(trainable)
        if trainable:
            selected.append((name, parameter))
    if not selected:
        raise RuntimeError("No parameters selected for adaptation")
    return selected


def model_inputs(batch, mode):
    video = batch["video"].cuda(non_blocking=True).permute(0, 2, 1, 3, 4)
    keyword = {"visual_mask": batch["visual_mask"].cuda(non_blocking=True)}
    if mode in ("multimodal", "shared_multimodal"):
        keyword.update(
            skeleton=batch["skeleton"].cuda(non_blocking=True),
            imu=batch["imu"].cuda(non_blocking=True),
            sensor_mask=batch["sensor_mask"].cuda(non_blocking=True),
        )
        if "radar" in batch:
            keyword["radar"] = batch["radar"].cuda(non_blocking=True)
    return video, keyword


def train_source_adapter(model, teacher, loader, mode, selected, args):
    model.eval()
    teacher.eval()
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in selected],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    history = []
    for epoch in range(args.epochs):
        loss_sum = 0.0
        correct = 0
        rows = 0
        for batch in loader:
            video, keyword = model_inputs(batch, mode)
            labels = batch["label"].cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                teacher_logits, _ = teacher(video, **keyword)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, auxiliary = model(video, **keyword)
                classification = functional.cross_entropy(logits, labels, label_smoothing=0.03)
                temperature = 2.0
                distillation = functional.kl_div(
                    functional.log_softmax(logits.float() / temperature, dim=1),
                    functional.softmax(teacher_logits.float() / temperature, dim=1),
                    reduction="batchmean",
                ) * temperature ** 2
                auxiliary_loss = torch.stack(
                    [functional.cross_entropy(values, labels) for values in auxiliary.values()]
                ).mean()
                loss = (
                    classification
                    + args.distillation_weight * distillation
                    + args.auxiliary_weight * auxiliary_loss
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_([parameter for _, parameter in selected], 2.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(labels)
            correct += int((logits.argmax(1) == labels).sum())
            rows += len(labels)
        scheduler.step()
        history.append(
            {
                "epoch": epoch + 1,
                "loss": loss_sum / rows,
                "accuracy": correct / rows,
                "lr": scheduler.get_last_lr()[0],
            }
        )
        print(json.dumps(history[-1]))
    return history


def main():
    args = parse_args()
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
    teacher = build_model(config, copy.deepcopy(state))
    selected = select_trainable(model, args.scope)
    initial = {name: parameter.detach().cpu().clone() for name, parameter in selected}

    records = get_records()
    source_indices = [
        record["index"]
        for record in records
        if record["split"] == "train" and record["user"] == args.source_user
    ]
    target_indices = [
        record["index"]
        for record in records
        if record["split"] == "train" and record["user"] == args.target_user
    ]
    if not source_indices or not target_indices:
        raise ValueError("Source or target user has no labeled development clips")
    source_dataset = R2Plus1DDataset(
        source_indices,
        args.cache_dir,
        training=True,
        include_sensors=mode in ("multimodal", "shared_multimodal"),
        include_radar=bool(config.get("use_radar", False)),
        canonical_pose=bool(config.get("canonical_pose", False)),
        frames=int(config.get("frames", 16)),
        image_size=int(config.get("image_size", 128)),
        photometric_jitter=0.05,
        horizontal_flip_probability=0.5,
    )
    generator = torch.Generator().manual_seed(args.seed)
    source_loader = DataLoader(
        source_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    history = train_source_adapter(model, teacher, source_loader, mode, selected, args)
    del teacher
    torch.cuda.empty_cache()

    target_loader = make_loader(target_indices, args, config, mode)
    if args.adabn != "none":
        adapt_batch_norm(model, target_loader, mode, passes=1, reset=args.adabn == "reset")
    rows = predict(model, target_loader, mode, args.tta_hflip, args.tta_crop_scale)
    ordered = {index: (label, values) for index, label, values, _ in rows}
    indices = np.asarray(target_indices, dtype=np.int64)
    labels = np.asarray([ordered[int(index)][0] for index in indices])
    probabilities = np.stack([ordered[int(index)][1] for index in indices])
    accuracy = float((probabilities.argmax(1) == labels).mean())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, probabilities=probabilities, labels=labels, indices=indices)
    adapter_path = args.adapter_output or args.output.with_suffix(".adapter.pt")
    adapter_path.parent.mkdir(parents=True, exist_ok=True)
    delta = {
        name: (parameter.detach().cpu() - initial[name]).half()
        for name, parameter in selected
    }
    torch.save(
        {
            "delta": delta,
            "source_user": args.source_user,
            "target_user": args.target_user,
            "scope": args.scope,
            "checkpoint": str(args.checkpoint),
        },
        adapter_path,
    )
    report = {
        "source_user": args.source_user,
        "target_user": args.target_user,
        "fold": args.fold,
        "source_rows": len(source_indices),
        "target_rows": len(target_indices),
        "target_accuracy": accuracy,
        "scope": args.scope,
        "trainable_parameters": sum(parameter.numel() for _, parameter in selected),
        "adapter_bytes": adapter_path.stat().st_size,
        "adabn": args.adabn,
        "history": history,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

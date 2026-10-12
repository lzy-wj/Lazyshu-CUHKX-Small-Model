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

from experiments.scene_context_v13.adapt_paired_source import checkpoint_state, model_inputs, select_trainable
from scripts.evaluate_r2plus1d_checkpoint import adapt_batch_norm, build_model, make_loader, predict
from src.data import get_indices, get_records
from src.r2plus1d_data import R2Plus1DDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pseudo-labels", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--target-user", type=int)
    parser.add_argument("--scope", choices=("heads", "fusion_last"), default="heads")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--confidence-threshold", type=float, default=0.7)
    parser.add_argument("--retention-weight", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--tta-hflip", action="store_true")
    parser.add_argument("--tta-crop-scale", type=float, default=1.0)
    parser.add_argument("--tta-upper-body-fraction", type=float, default=1.0)
    parser.add_argument("--train-upper-body-fraction", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.5 <= args.train_upper_body_fraction <= 1:
        raise ValueError("Train upper-body fraction must be in [0.5, 1]")
    if not 0.5 <= args.tta_upper_body_fraction <= 1:
        raise ValueError("TTA upper-body fraction must be in [0.5, 1]")
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
    selected_parameters = select_trainable(model, args.scope)
    records = get_records()
    _, target_indices = get_indices(records, args.fold)
    if args.target_user is not None:
        available_users = {records[index]["user"] for index in target_indices}
        if args.target_user not in available_users:
            raise ValueError(f"Target user {args.target_user} is not in Fold {args.fold}")
        target_indices = [
            index for index in target_indices if records[index]["user"] == args.target_user
        ]
    archive = np.load(args.pseudo_labels)
    positions = {int(index): position for position, index in enumerate(archive["indices"])}
    pseudo_labels = torch.full((len(records),), -1, dtype=torch.long, device="cuda")
    selected_rows = torch.zeros(len(records), dtype=torch.bool, device="cuda")
    for index in target_indices:
        position = positions[index]
        pseudo_labels[index] = int(archive["pseudo_labels"][position])
        selected_rows[index] = bool(
            archive["paired"][position]
            or archive["confidence"][position] >= args.confidence_threshold
        )
    dataset = R2Plus1DDataset(
        target_indices,
        args.cache_dir,
        training=True,
        include_sensors=mode in ("multimodal", "shared_multimodal"),
        include_radar=bool(config.get("use_radar", False)),
        canonical_pose=bool(config.get("canonical_pose", False)),
        frames=args.frames or int(config.get("frames", 16)),
        image_size=args.image_size or int(config.get("image_size", 128)),
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
        [parameter for _, parameter in selected_parameters], lr=args.lr, weight_decay=1e-4
    )
    history = []
    model.eval()
    teacher.eval()
    for epoch in range(args.epochs):
        loss_sum = 0.0
        rows = 0
        for batch in loader:
            indices = batch["index"].cuda(non_blocking=True)
            active = selected_rows[indices]
            if not active.any():
                continue
            video, keyword = model_inputs(batch, mode)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                teacher_logits, _ = teacher(video, **keyword)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(video, **keyword)
                pseudo_loss = functional.cross_entropy(logits[active], pseudo_labels[indices][active])
                retention = functional.kl_div(
                    functional.log_softmax(logits.float(), dim=1),
                    functional.softmax(teacher_logits.float(), dim=1),
                    reduction="batchmean",
                )
                if args.train_upper_body_fraction < 1:
                    upper_height = round(video.shape[-2] * args.train_upper_body_fraction)
                    upper = functional.interpolate(
                        video[..., :upper_height, :],
                        size=video.shape[-3:],
                        mode="trilinear",
                        align_corners=False,
                    )
                    with torch.no_grad():
                        teacher_upper_logits, _ = teacher(upper, **keyword)
                    upper_logits, _ = model(upper, **keyword)
                    upper_pseudo_loss = functional.cross_entropy(
                        upper_logits[active], pseudo_labels[indices][active]
                    )
                    upper_retention = functional.kl_div(
                        functional.log_softmax(upper_logits.float(), dim=1),
                        functional.softmax(teacher_upper_logits.float(), dim=1),
                        reduction="batchmean",
                    )
                    pseudo_loss = 0.5 * (pseudo_loss + upper_pseudo_loss)
                    retention = 0.5 * (retention + upper_retention)
                loss = pseudo_loss + args.retention_weight * retention
            loss.backward()
            torch.nn.utils.clip_grad_norm_([parameter for _, parameter in selected_parameters], 2.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * int(active.sum())
            rows += int(active.sum())
        history.append({"epoch": epoch + 1, "loss": loss_sum / max(rows, 1), "rows": rows})
        print(json.dumps(history[-1]))
    del teacher
    target_loader = make_loader(target_indices, args, config, mode)
    adapt_batch_norm(model, target_loader, mode, passes=1, reset=True)
    predicted = predict(
        model,
        target_loader,
        mode,
        args.tta_hflip,
        args.tta_crop_scale,
        upper_body_fraction=args.tta_upper_body_fraction,
    )
    ordered = {index: (label, values) for index, label, values, _ in predicted}
    indices = np.asarray(target_indices, dtype=np.int64)
    labels = np.asarray([ordered[int(index)][0] for index in indices])
    probabilities = np.stack([ordered[int(index)][1] for index in indices])
    report = {
        "accuracy": float((probabilities.argmax(1) == labels).mean()),
        "fold": args.fold,
        "target_user": args.target_user,
        "scope": args.scope,
        "epochs": args.epochs,
        "lr": args.lr,
        "confidence_threshold": args.confidence_threshold,
        "image_size": args.image_size or int(config.get("image_size", 128)),
        "tta_upper_body_fraction": args.tta_upper_body_fraction,
        "train_upper_body_fraction": args.train_upper_body_fraction,
        "selected_pseudo_rows": int(selected_rows.sum()),
        "trainable_parameters": sum(parameter.numel() for _, parameter in selected_parameters),
        "history": history,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, probabilities=probabilities, labels=labels, indices=indices)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

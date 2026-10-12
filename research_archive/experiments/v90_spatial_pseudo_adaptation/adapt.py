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

from experiments.v80_spatial_masked_encoder.model import build_from_source
from scripts.evaluate_r2plus1d_checkpoint import adapt_batch_norm, make_loader, model_inputs, predict
from src.data import get_indices, get_records
from src.r2plus1d_data import R2Plus1DDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pseudo-labels", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--confidence-threshold", type=float, default=0.6)
    parser.add_argument("--retention-weight", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tta-hflip", action="store_true")
    parser.add_argument("--tta-crop-scale", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter-output", type=Path)
    return parser.parse_args()


def build_model(checkpoint):
    config = checkpoint["config"]
    source = Path(config["init_checkpoint"])
    model, _ = build_from_source(
        source,
        dimension=int(config["dimension"]),
        depth=int(config["depth"]),
        heads=int(config["heads"]),
        feedforward_multiplier=int(config["feedforward_multiplier"]),
        residual_scale=float(config["residual_scale"]),
        reconstruction_mask_probability=float(config["reconstruction_mask_probability"]),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return model


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(checkpoint)
    teacher = copy.deepcopy(model)
    selected = []
    for name, parameter in model.named_parameters():
        trainable = name.startswith("residual_head.")
        parameter.requires_grad_(trainable)
        if trainable:
            selected.append((name, parameter))
    if not selected:
        raise RuntimeError("No residual-head parameters selected")
    model.cuda().eval()
    teacher.cuda().eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    records = get_records()
    _, target_indices = get_indices(records, args.fold)
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

    train_dataset = R2Plus1DDataset(
        target_indices,
        args.cache_dir,
        training=True,
        include_sensors=True,
        include_radar=False,
        frames=args.frames,
        image_size=args.image_size,
        photometric_jitter=0.05,
        horizontal_flip_probability=0.5,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in selected],
        lr=args.lr,
        weight_decay=1e-4,
    )
    history = []
    for epoch in range(args.epochs):
        loss_sum = 0.0
        rows = 0
        for batch in train_loader:
            indices = batch["index"].cuda(non_blocking=True)
            active = selected_rows[indices]
            if not active.any():
                continue
            video, keyword = model_inputs(batch, "shared_multimodal")
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
                loss = pseudo_loss + args.retention_weight * retention
            loss.backward()
            torch.nn.utils.clip_grad_norm_([parameter for _, parameter in selected], 2.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * int(active.sum())
            rows += int(active.sum())
        value = {"epoch": epoch + 1, "loss": loss_sum / max(rows, 1), "rows": rows}
        history.append(value)
        print(json.dumps(value), flush=True)

    del teacher
    target_loader = make_loader(target_indices, args, checkpoint["config"], "shared_multimodal")
    adapt_batch_norm(model, target_loader, "shared_multimodal", passes=1, reset=True)
    predicted = predict(
        model,
        target_loader,
        "shared_multimodal",
        args.tta_hflip,
        args.tta_crop_scale,
    )
    ordered = {index: (label, values) for index, label, values, _ in predicted}
    indices = np.asarray(target_indices, dtype=np.int64)
    labels = np.asarray([ordered[int(index)][0] for index in indices])
    probabilities = np.stack([ordered[int(index)][1] for index in indices])
    report = {
        "accuracy": float((probabilities.argmax(1) == labels).mean()),
        "fold": args.fold,
        "epochs": args.epochs,
        "lr": args.lr,
        "confidence_threshold": args.confidence_threshold,
        "retention_weight": args.retention_weight,
        "image_size": args.image_size,
        "selected_pseudo_rows": int(selected_rows.sum()),
        "trainable_parameters": sum(parameter.numel() for _, parameter in selected),
        "history": history,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, probabilities=probabilities, labels=labels, indices=indices)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    if args.adapter_output:
        args.adapter_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": {name: parameter.detach().cpu() for name, parameter in selected}},
            args.adapter_output,
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

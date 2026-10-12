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
from src.data import get_records
from src.r2plus1d_data import R2Plus1DDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pseudo-labels", type=Path, required=True)
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter-output", type=Path)
    return parser.parse_args()


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
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
    initial = {
        name: parameter.detach().cpu().clone()
        for name, parameter in selected_parameters
    }

    records = get_records()
    target_indices = [record["index"] for record in records if record["split"] == "test"]
    archive = np.load(args.pseudo_labels)
    if not np.array_equal(archive["indices"], np.asarray(target_indices)):
        raise ValueError("Pseudo labels are not aligned with the complete test split")
    pseudo_labels = torch.full((len(records),), -1, dtype=torch.long, device="cuda")
    selected_rows = torch.zeros(len(records), dtype=torch.bool, device="cuda")
    for position, index in enumerate(target_indices):
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
        [parameter for _, parameter in selected_parameters],
        lr=args.lr,
        weight_decay=1e-4,
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
                loss = pseudo_loss + args.retention_weight * retention
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for _, parameter in selected_parameters],
                2.0,
            )
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
    ordered = {index: values for index, _, values, _ in predicted}
    indices = np.asarray(target_indices, dtype=np.int64)
    probabilities = np.stack([ordered[int(index)] for index in indices])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, probabilities=probabilities, indices=indices)
    adapter_path = args.adapter_output or args.output.with_suffix(".adapter.pt")
    adapter_path.parent.mkdir(parents=True, exist_ok=True)
    delta = {
        name: (parameter.detach().cpu() - initial[name]).half()
        for name, parameter in selected_parameters
    }
    torch.save(
        {
            "delta": delta,
            "scope": args.scope,
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "pseudo_labels_sha256": file_sha256(args.pseudo_labels),
            "confidence_threshold": args.confidence_threshold,
            "epochs": args.epochs,
            "lr": args.lr,
            "retention_weight": args.retention_weight,
            "tta_upper_body_fraction": args.tta_upper_body_fraction,
        },
        adapter_path,
    )
    report = {
        "checkpoint": str(args.checkpoint),
        "rows": len(indices),
        "selected_pseudo_rows": int(selected_rows.sum()),
        "scope": args.scope,
        "trainable_parameters": sum(parameter.numel() for _, parameter in selected_parameters),
        "adapter": str(adapter_path),
        "adapter_bytes": adapter_path.stat().st_size,
        "mean_confidence": float(probabilities.max(axis=1).mean()),
        "tta_upper_body_fraction": args.tta_upper_body_fraction,
        "history": history,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

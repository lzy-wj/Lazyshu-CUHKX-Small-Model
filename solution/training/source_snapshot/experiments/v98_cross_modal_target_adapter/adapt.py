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

from experiments.v97_protected_visual_adapters.model import build_from_source
from scripts.evaluate_r2plus1d_checkpoint import adapt_batch_norm, make_loader, model_inputs, predict
from src.data import get_indices, get_records
from src.r2plus1d_data import R2Plus1DDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pseudo-labels", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--target-domain", type=int, choices=(0, 1, 2, 3))
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=HERE / "outputs")
    parser.add_argument("--bottleneck", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--confidence-threshold", type=float, default=0.6)
    parser.add_argument("--retention-weight", type=float, default=0.25)
    parser.add_argument("--stream-distillation-weight", type=float, default=0.0)
    parser.add_argument("--stream-temperature", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--tta-hflip", action="store_true")
    parser.add_argument("--tta-crop-scale", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--swanlab-mode", choices=("cloud", "local", "disabled"), default="cloud")
    return parser.parse_args()


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stream_distillation_loss(stream_logits, teacher_logits, visual_mask, temperature):
    teacher = functional.softmax(teacher_logits.float() / temperature, dim=1).unsqueeze(1)
    student = functional.log_softmax(stream_logits.float() / temperature, dim=2)
    losses = functional.kl_div(student, teacher.expand_as(student), reduction="none").sum(dim=2)
    mask = visual_mask.float()
    return (losses * mask).sum() / mask.sum().clamp_min(1) * temperature ** 2


def main():
    args = parse_args()
    if args.stream_temperature <= 0:
        raise ValueError("Stream temperature must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model, teacher, config = build_from_source(
        args.checkpoint,
        bottleneck=args.bottleneck,
        output_adapters=True,
    )
    if int(config["fold"]) != args.fold:
        raise ValueError("Initialization checkpoint and target fold disagree")
    model.cuda().eval()
    teacher.cuda().eval()
    selected = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    initial = {name: parameter.detach().cpu().clone() for name, parameter in selected}

    records = get_records()
    _, target_indices = get_indices(records, args.fold)
    if args.target_domain is not None:
        paired_users = ({8, 18}, {7, 17})[args.fold]
        target_indices = [
            index
            for index in target_indices
            if int(records[index]["user"] > 15)
            + 2 * int(records[index]["user"] in paired_users)
            == args.target_domain
        ]
        if not target_indices:
            raise ValueError("Target-domain filter selected no validation rows")
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

    train_loader = DataLoader(
        R2Plus1DDataset(
            target_indices,
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
        [parameter for _, parameter in selected],
        lr=args.lr,
        weight_decay=1e-4,
    )
    destination = args.output_root / args.name
    destination.mkdir(parents=True, exist_ok=False)
    run_config = {
        **vars(args),
        "checkpoint": str(args.checkpoint),
        "pseudo_labels": str(args.pseudo_labels),
        "cache_dir": str(args.cache_dir),
        "output_root": str(args.output_root),
        "trainable_parameters": sum(parameter.numel() for _, parameter in selected),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "pseudo_labels_sha256": file_sha256(args.pseudo_labels),
        "architecture": (
            "frozen CNN/RNN/Transformer teacher and backbone with zero-initialized "
            "modality-specific spatiotemporal CNN adapters"
        ),
    }
    serializable = {key: str(value) if isinstance(value, Path) else value for key, value in run_config.items()}
    (destination / "config.json").write_text(json.dumps(serializable, indent=2) + "\n")

    import swanlab

    run = swanlab.init(
        project="cuhkx-small-multimodal",
        name=args.name,
        config=serializable,
        public=False,
        mode=args.swanlab_mode,
        log_dir=str(HERE / "swanlog"),
        description="Target-domain multimodal teacher to visual-stream adapter distillation.",
    )
    history = []
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        sums = {"loss": 0.0, "pseudo": 0.0, "retention": 0.0, "stream": 0.0}
        rows = 0
        for raw_batch in train_loader:
            indices = raw_batch["index"].cuda(non_blocking=True)
            active = selected_rows[indices]
            if not active.any():
                continue
            video, keyword = model_inputs(raw_batch, "shared_multimodal")
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                teacher_logits, _ = teacher(video, **keyword)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, details = model(video, **keyword)
                pseudo = functional.cross_entropy(logits[active], pseudo_labels[indices][active])
                retention = functional.kl_div(
                    functional.log_softmax(logits.float(), dim=1),
                    functional.softmax(teacher_logits.float(), dim=1),
                    reduction="batchmean",
                )
                stream = stream_distillation_loss(
                    details["video_streams"],
                    teacher_logits,
                    keyword["visual_mask"],
                    args.stream_temperature,
                )
                loss = (
                    pseudo
                    + args.retention_weight * retention
                    + args.stream_distillation_weight * stream
                )
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite V98 loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([parameter for _, parameter in selected], 2.0)
            optimizer.step()
            count = int(active.sum())
            rows += count
            sums["loss"] += float(loss.detach()) * count
            sums["pseudo"] += float(pseudo.detach()) * count
            sums["retention"] += float(retention.detach()) * count
            sums["stream"] += float(stream.detach()) * count
        values = {
            "epoch": epoch,
            "train/loss": sums["loss"] / rows,
            "train/pseudo": sums["pseudo"] / rows,
            "train/retention": sums["retention"] / rows,
            "train/stream_distillation": sums["stream"] / rows,
            "train/selected_rows": rows,
            "system/peak_cuda_mb": torch.cuda.max_memory_allocated() / 1e6,
        }
        history.append(values)
        with (destination / "history.jsonl").open("a") as handle:
            handle.write(json.dumps(values) + "\n")
        print(json.dumps(values), flush=True)
        swanlab.log({key: value for key, value in values.items() if key != "epoch"}, step=epoch)

    target_loader = make_loader(target_indices, args, config, "shared_multimodal")
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
    np.savez(destination / "validation.npz", probabilities=probabilities, labels=labels, indices=indices)
    adapter = {
        name: (parameter.detach().cpu() - initial[name]).half()
        for name, parameter in selected
    }
    torch.save({"delta": adapter, "config": serializable}, destination / "adapter.pt")
    report = {
        "accuracy": float((probabilities.argmax(axis=1) == labels).mean()),
        "correct": int((probabilities.argmax(axis=1) == labels).sum()),
        "rows": len(labels),
        "selected_pseudo_rows": int(selected_rows.sum()),
        "trainable_parameters": serializable["trainable_parameters"],
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

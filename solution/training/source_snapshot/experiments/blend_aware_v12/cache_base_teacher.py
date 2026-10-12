#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data import ClipDataset, get_indices, get_records, prepare_batch
from src.models import MultiModalModel
from src.r2plus1d_models import dequantize_state_dict


BASE_WEIGHTS = (0.83, 0.17)
BASE_TEMPERATURES = (0.9736012920941253, 1.1338148770999141)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--partition", choices=("train", "valid"), default="train")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def build_model(config):
    return MultiModalModel(
        config["modalities"],
        width=config["width"],
        spatial_pool=config.get("spatial_pool", 1),
        backbone=config.get("backbone", "scratch"),
        skeleton_backbone=config.get("skeleton_backbone", "tcn"),
        temporal_backbone=config.get("temporal_backbone", "tcn"),
        temporal_shift=config.get("temporal_shift", False),
        kinetics_source=config.get("kinetics_source", "s"),
        fusion=config.get("fusion", "concat"),
    )


@torch.inference_mode()
def predict(path, indices, temperature, args):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if int(config["fold"]) != args.fold:
        raise ValueError(f"Teacher checkpoint fold mismatch: {path}")
    model = build_model(config)
    state = (
        dequantize_state_dict(checkpoint["model_int8"])
        if "model_int8" in checkpoint else checkpoint["model"]
    )
    model.load_state_dict(state, strict=True)
    model = model.cuda().eval()
    dataset = ClipDataset(
        indices,
        config["modalities"],
        frames=config["frames"],
        canonical_pose=config.get("canonical_pose", False),
        image_size=config.get("image_size", 96),
        thermal_view=config.get("thermal_view", "full"),
        imu_representation=config.get("imu_representation", "raw"),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    rows = []
    for batch in loader:
        batch = prepare_batch(batch, torch.device("cuda"))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(batch)
        rows.append((logits.float() / temperature).softmax(dim=1).cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(rows), config, checkpoint.get("epoch")


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Teacher caching requires CUDA")
    records = get_records()
    train_indices, valid_indices = get_indices(records, args.fold)
    indices = np.asarray(train_indices if args.partition == "train" else valid_indices, dtype=np.int64)
    paths = [
        ROOT / f"outputs/checkpoints/v3_x3d_f{args.fold}/best.pt",
        ROOT / f"outputs/checkpoints/v2_statistics_f{args.fold}/best.pt",
    ]
    components = []
    manifests = []
    for path, temperature in zip(paths, BASE_TEMPERATURES):
        probabilities, config, epoch = predict(path, indices, temperature, args)
        components.append(probabilities)
        manifests.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "epoch": epoch,
                "temperature": temperature,
                "modalities": config["modalities"],
            }
        )
    probabilities = sum(weight * values for weight, values in zip(BASE_WEIGHTS, components))
    labels = np.asarray([records[int(index)]["label"] for index in indices], dtype=np.int64)
    if probabilities.shape != (len(indices), 40) or not np.isfinite(probabilities).all():
        raise RuntimeError("Invalid teacher probabilities")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        probabilities=probabilities.astype(np.float32),
        per_model=np.stack(components).astype(np.float32),
        labels=labels,
        indices=indices,
        weights=np.asarray(BASE_WEIGHTS),
        temperatures=np.asarray(BASE_TEMPERATURES),
    )
    report = {
        "fold": args.fold,
        "partition": args.partition,
        "rows": len(indices),
        "accuracy": float((probabilities.argmax(axis=1) == labels).mean()),
        "mean_target_probability": float(probabilities[np.arange(len(labels)), labels].mean()),
        "mean_confidence": float(probabilities.max(axis=1).mean()),
        "weights": BASE_WEIGHTS,
        "models": manifests,
        "output": str(args.output),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

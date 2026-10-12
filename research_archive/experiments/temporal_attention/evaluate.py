#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional
from torch.utils.data import DataLoader


EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EXPERIMENT_ROOT))

from model import TemporalAttentionR2Plus1DClassifier
from src.data import get_indices, get_records
from src.r2plus1d_data import R2Plus1DDataset
from src.r2plus1d_models import dequantize_state_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tta-hflip", action="store_true")
    parser.add_argument("--tta-crop-scale", type=float, default=1.0)
    return parser.parse_args()


def inputs(batch):
    video = batch["video"].cuda(non_blocking=True).permute(0, 2, 1, 3, 4)
    keyword = {
        "visual_mask": batch["visual_mask"].cuda(non_blocking=True),
        "skeleton": batch["skeleton"].cuda(non_blocking=True),
        "imu": batch["imu"].cuda(non_blocking=True),
        "sensor_mask": batch["sensor_mask"].cuda(non_blocking=True),
    }
    return video, keyword


@torch.no_grad()
def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    state = dequantize_state_dict(checkpoint["model_int8"]) if "model_int8" in checkpoint else checkpoint["model"]
    model = TemporalAttentionR2Plus1DClassifier(
        weights=None,
        dropout=float(config["dropout"]),
        input_channels=int(config["input_channels"]),
    ).cuda()
    model.load_state_dict(state, strict=True)
    model.eval()
    _, indices = get_indices(get_records(), 0)
    dataset = R2Plus1DDataset(
        indices,
        Path(config["cache_dir"]),
        training=False,
        include_sensors=True,
        frames=int(config["frames"]),
        image_size=int(config["image_size"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    all_probabilities = []
    all_labels = []
    all_indices = []
    for batch in loader:
        video, keyword = inputs(batch)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(video, **keyword)
            views = [logits]
            if args.tta_hflip:
                views.append(model(torch.flip(video, dims=(-1,)), **keyword)[0])
            if args.tta_crop_scale < 1:
                side = round(video.shape[-1] * args.tta_crop_scale)
                top = (video.shape[-2] - side) // 2
                left = (video.shape[-1] - side) // 2
                cropped = functional.interpolate(
                    video[..., top:top + side, left:left + side],
                    size=video.shape[-3:],
                    mode="trilinear",
                    align_corners=False,
                )
                views.append(model(cropped, **keyword)[0])
                if args.tta_hflip:
                    views.append(model(torch.flip(cropped, dims=(-1,)), **keyword)[0])
            probabilities = torch.stack(views).mean(0).float().softmax(1).cpu().numpy()
        all_probabilities.append(probabilities)
        all_labels.append(batch["label"].numpy())
        all_indices.append(batch["index"].numpy())
    probabilities = np.concatenate(all_probabilities)
    labels = np.concatenate(all_labels)
    ordered_indices = np.concatenate(all_indices)
    available = np.load(REPO_ROOT / "data/processed/r2plus1d_yolo_32_128/available.npy").astype(bool)[ordered_indices]
    metrics = {
        "accuracy": float((probabilities.argmax(1) == labels).mean()),
        "available_accuracy": float((probabilities[available].argmax(1) == labels[available]).mean()),
        "coverage": float(available.mean()),
        "fold": 0,
        "frames": int(config["frames"]),
        "tta_hflip": args.tta_hflip,
        "tta_crop_scale": args.tta_crop_scale,
        "checkpoint": str(args.checkpoint),
        "parameters": int(config["parameters"]),
        "baseline_parameters": int(config["baseline_parameters"]),
        "parameter_delta": int(config["parameter_delta"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, probabilities=probabilities, labels=labels, indices=ordered_indices)
    args.output.with_suffix(".json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

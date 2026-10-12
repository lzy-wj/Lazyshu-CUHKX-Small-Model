#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_r2plus1d_checkpoint import (
    build_model,
    make_loader,
    model_inputs,
)
from src.data import get_indices, get_records
from src.r2plus1d_models import dequantize_state_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--crop-fraction", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=128)
    return parser.parse_args()


def checkpoint_state(checkpoint):
    for key in ("model_int4", "model_int7", "model_int6", "model_int8"):
        if key in checkpoint:
            return dequantize_state_dict(checkpoint[key])
    return checkpoint["model"]


def motion_crop(video, fraction):
    batch, channels, steps, height, width = video.shape
    side = round(min(height, width) * fraction)
    difference = torch.diff(video.float(), dim=2).abs()
    scale = difference.mean(dim=(2, 3, 4), keepdim=True).clamp_min(1e-4)
    saliency = (difference / scale).mean(dim=(1, 2))
    saliency = functional.avg_pool2d(
        saliency.unsqueeze(1), kernel_size=15, stride=1, padding=7
    ).squeeze(1)
    saliency[:, : round(height * 0.08)] = 0
    saliency[:, round(height * 0.90) :] = 0
    threshold = torch.quantile(saliency.flatten(1), 0.85, dim=1)
    weights = (saliency - threshold[:, None, None]).clamp_min(0)
    y_grid = torch.arange(height, device=video.device).view(1, height, 1)
    x_grid = torch.arange(width, device=video.device).view(1, 1, width)
    mass = weights.sum(dim=(1, 2)).clamp_min(1e-6)
    center_y = ((weights * y_grid).sum(dim=(1, 2)) / mass).round().long()
    center_x = ((weights * x_grid).sum(dim=(1, 2)) / mass).round().long()
    crops = []
    for position in range(batch):
        top = int((center_y[position] - side // 2).clamp(0, height - side))
        left = int((center_x[position] - side // 2).clamp(0, width - side))
        crops.append(video[position : position + 1, :, :, top : top + side, left : left + side])
    cropped = torch.cat(crops)
    return functional.interpolate(
        cropped,
        size=(steps, height, width),
        mode="trilinear",
        align_corners=False,
    )


@torch.no_grad()
def main():
    args = parse_args()
    if not 0.4 <= args.crop_fraction <= 0.9:
        raise ValueError("--crop-fraction must be in [0.4, 0.9]")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"].copy()
    if int(config["fold"]) != args.fold:
        raise ValueError("Checkpoint fold does not match --fold")
    model = build_model(config, checkpoint_state(checkpoint))
    records = get_records()
    _, indices = get_indices(records, args.fold)
    loader = make_loader(indices, args, config, config["mode"])
    probabilities = []
    ordered_indices = []
    labels = []
    for batch in loader:
        video, keyword = model_inputs(batch, config["mode"])
        cropped = motion_crop(video, args.crop_fraction)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(cropped, **keyword)
        probabilities.append(logits.float().softmax(dim=1).cpu().numpy())
        ordered_indices.extend(batch["index"].numpy().tolist())
        labels.extend(batch["label"].numpy().tolist())
    payload = {
        "probabilities": np.concatenate(probabilities),
        "indices": np.asarray(ordered_indices, dtype=np.int64),
        "labels": np.asarray(labels, dtype=np.int64),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **payload)
    report = {
        "fold": args.fold,
        "rows": len(ordered_indices),
        "crop_fraction": args.crop_fraction,
        "raw_correct": int(
            (payload["probabilities"].argmax(axis=1) == payload["labels"]).sum()
        ),
        "output": str(args.output),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()


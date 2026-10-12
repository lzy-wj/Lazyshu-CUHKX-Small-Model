#!/usr/bin/env python3
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_r2plus1d_checkpoint import adapt_batch_norm, build_model, make_loader, predict
from src.data import get_records
from src.r2plus1d_models import dequantize_state_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tta-hflip", action="store_true")
    parser.add_argument("--tta-crop-scale", type=float, default=0.75)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def checkpoint_state(checkpoint):
    for key in ("model_int8", "model_int7", "model_int6", "model_int4"):
        if key in checkpoint:
            return dequantize_state_dict(checkpoint[key])
    return checkpoint["model"]


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    mode = config["mode"]
    records = get_records()
    indices = [record["index"] for record in records if record["split"] == "test"]
    model = build_model(config, copy.deepcopy(checkpoint_state(checkpoint)))
    loader = make_loader(indices, args, config, mode)
    ignore_radar = bool(config.get("ignore_radar", False))
    adapt_batch_norm(model, loader, mode, passes=1, reset=True, ignore_radar=ignore_radar)
    predicted = predict(
        model,
        loader,
        mode,
        args.tta_hflip,
        args.tta_crop_scale,
        ignore_radar=ignore_radar,
    )
    ordered = {index: (values, branches) for index, _, values, branches in predicted}
    ordered_indices = np.asarray(indices, dtype=np.int64)
    probabilities = np.stack([ordered[int(index)][0] for index in ordered_indices])
    branches = {
        f"probabilities_{name}": np.stack([ordered[int(index)][1][name] for index in ordered_indices])
        for name in next(iter(ordered.values()))[1]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, probabilities=probabilities, indices=ordered_indices, **branches)
    report = {
        "checkpoint": str(args.checkpoint),
        "rows": len(indices),
        "branches": sorted(branches),
        "adabn": "global_reset",
        "tta_hflip": args.tta_hflip,
        "tta_crop_scale": args.tta_crop_scale,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.r2plus1d_models import quantize_state_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--right-weight", type=float, default=0.75)
    parser.add_argument("--visual-new-weight", type=float, required=True)
    parser.add_argument("--other-new-weight", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "model" not in checkpoint:
        raise ValueError(f"FP model state required: {path}")
    return checkpoint


def interpolate(left, right, right_weight):
    if left.keys() != right.keys():
        raise ValueError("State dictionaries differ")
    output = {}
    for name in left:
        if left[name].shape != right[name].shape:
            raise ValueError(f"Tensor shape mismatch: {name}")
        output[name] = (
            left[name].float().lerp(right[name].float(), right_weight).half()
            if left[name].is_floating_point() else right[name].clone()
        )
    return output


def main():
    args = parse_args()
    for value in (args.right_weight, args.visual_new_weight, args.other_new_weight):
        if not 0 <= value <= 1:
            raise ValueError("Soup weights must be in [0, 1]")
    left = load(args.left)
    old = load(args.old)
    new = load(args.new)
    old_state = old["model"]
    new_state = new["model"]
    if old_state.keys() != new_state.keys():
        raise ValueError("Old and new model states differ")
    right_state = {}
    for name in old_state:
        new_weight = (
            args.visual_new_weight
            if name.startswith("video.") or name.startswith("video_head.")
            else args.other_new_weight
        )
        right_state[name] = (
            old_state[name].float().lerp(new_state[name].float(), new_weight).half()
            if old_state[name].is_floating_point() else new_state[name].clone()
        )
    state = interpolate(left["model"], right_state, args.right_weight)
    quantized, tensor_bytes = quantize_state_dict(state)
    config = dict(new["config"])
    config["module_soup"] = {
        "left": str(args.left),
        "old": str(args.old),
        "new": str(args.new),
        "right_weight": args.right_weight,
        "visual_new_weight": args.visual_new_weight,
        "other_new_weight": args.other_new_weight,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_int8": quantized,
            "config": config,
            "epoch": None,
            "metrics": {},
            "tensor_bytes": tensor_bytes,
        },
        args.output,
    )
    report = {
        "output": str(args.output),
        "bytes": args.output.stat().st_size,
        "tensor_bytes": tensor_bytes,
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "module_soup": config["module_soup"],
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

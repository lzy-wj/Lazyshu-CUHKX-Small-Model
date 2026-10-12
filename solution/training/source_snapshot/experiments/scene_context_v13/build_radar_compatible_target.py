#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.r2plus1d_models import dequantize_state_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def state(checkpoint):
    for key in ("model_int8", "model_int7", "model_int6", "model_int4"):
        if key in checkpoint:
            return dequantize_state_dict(checkpoint[key])
    return checkpoint["model"]


def main():
    args = parse_args()
    base_checkpoint = torch.load(args.base, map_location="cpu", weights_only=False)
    target_checkpoint = torch.load(args.target, map_location="cpu", weights_only=False)
    base = state(base_checkpoint)
    target = state(target_checkpoint)
    expanded = {}
    copied_target = []
    copied_base = []
    for name, base_value in base.items():
        if name == "modality_embedding":
            value = base_value.clone()
            if target[name].shape[1] + 1 != value.shape[1]:
                raise ValueError("Expected the target to omit exactly one radar modality embedding")
            value[:, :target[name].shape[1]] = target[name].to(value.dtype)
            expanded[name] = value
            copied_target.append(name)
        elif name in target and target[name].shape == base_value.shape:
            expanded[name] = target[name].to(base_value.dtype)
            copied_target.append(name)
        else:
            expanded[name] = base_value.clone()
            copied_base.append(name)
    unexpected = set(target) - set(base)
    if unexpected:
        raise ValueError(f"Target tensors are absent from the radar-compatible base: {unexpected}")
    config = base_checkpoint["config"].copy()
    config.update(
        {
            "name": f"{target_checkpoint['config']['name']}_radar_compatible",
            "visual_strategy": target_checkpoint["config"].get("visual_strategy", "sequential"),
            "ignore_radar": True,
        }
    )
    output = {
        "model": {
            name: value.half() if value.is_floating_point() else value
            for name, value in expanded.items()
        },
        "config": config,
        "source_target": str(args.target),
        "copied_target_tensors": copied_target,
        "copied_base_tensors": copied_base,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(
        {
            "output": str(args.output),
            "bytes": args.output.stat().st_size,
            "target_tensors": len(copied_target),
            "base_tensors": len(copied_base),
        }
    )


if __name__ == "__main__":
    main()

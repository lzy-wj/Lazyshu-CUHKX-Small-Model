#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.r2plus1d_models import dequantize_state_dict, quantize_state_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint", type=Path, required=True)
    parser.add_argument("--sequential", type=Path, required=True)
    parser.add_argument("--joint-weight", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def state(checkpoint):
    if "model" in checkpoint:
        return checkpoint["model"]
    if "model_int8" in checkpoint:
        return dequantize_state_dict(checkpoint["model_int8"])
    raise ValueError("Cross-strategy soup requires FP or INT8 source checkpoints")


def main():
    args = parse_args()
    if not 0 <= args.joint_weight <= 1:
        raise ValueError("Joint weight must be in [0, 1]")
    joint = torch.load(args.joint, map_location="cpu", weights_only=False)
    sequential = torch.load(args.sequential, map_location="cpu", weights_only=False)
    joint_state = state(joint)
    sequential_state = state(sequential)
    if joint_state.keys() != sequential_state.keys():
        raise ValueError("Source state dictionaries differ")
    config = dict(joint["config"])
    if config.get("visual_strategy") != "joint":
        raise ValueError("The deployment checkpoint must use joint visual encoding")
    if sequential["config"].get("visual_strategy", "sequential") != "sequential":
        raise ValueError("The second source must use sequential visual encoding")
    mixed = {}
    for name, joint_value in joint_state.items():
        sequential_value = sequential_state[name]
        if joint_value.shape != sequential_value.shape:
            raise ValueError(f"Tensor shape mismatch: {name}")
        mixed[name] = (
            sequential_value.float().lerp(joint_value.float(), args.joint_weight).half()
            if joint_value.is_floating_point()
            else joint_value.clone()
        )
    quantized, tensor_bytes = quantize_state_dict(mixed)
    sources = [
        {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "weight": weight,
        }
        for path, weight in (
            (args.joint, args.joint_weight),
            (args.sequential, 1 - args.joint_weight),
        )
    ]
    config["cross_strategy_soup"] = sources
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
        "checkpoint_bytes": args.output.stat().st_size,
        "tensor_bytes": tensor_bytes,
        "sources": sources,
        "inference_visual_strategy": "joint",
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

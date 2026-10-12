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
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--weights", type=float, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def state(checkpoint):
    if "model" in checkpoint:
        return checkpoint["model"]
    if "model_int8" in checkpoint:
        return dequantize_state_dict(checkpoint["model_int8"])
    raise ValueError("Unsupported checkpoint format")


def main():
    args = parse_args()
    if len(args.checkpoints) != len(args.weights) or any(weight < 0 for weight in args.weights):
        raise ValueError("Supply one nonnegative weight per checkpoint")
    total = sum(args.weights)
    if total <= 0:
        raise ValueError("Soup weights must have positive total")
    weights = [weight / total for weight in args.weights]
    checkpoints = [torch.load(path, map_location="cpu", weights_only=False) for path in args.checkpoints]
    states = [state(checkpoint) for checkpoint in checkpoints]
    if any(current.keys() != states[0].keys() for current in states[1:]):
        raise ValueError("Soup state dictionaries differ")
    mixed = {}
    for name in states[0]:
        values = [current[name] for current in states]
        if any(value.shape != values[0].shape for value in values[1:]):
            raise ValueError(f"Soup tensor shape mismatch: {name}")
        mixed[name] = (
            sum(weight * value.float() for weight, value in zip(weights, values)).half()
            if values[0].is_floating_point() else values[-1].clone()
        )
    quantized, tensor_bytes = quantize_state_dict(mixed)
    config = dict(checkpoints[-1]["config"])
    config["weighted_checkpoint_soup"] = [
        {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "weight": weight,
        }
        for path, weight in zip(args.checkpoints, weights)
    ]
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
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "tensor_bytes": tensor_bytes,
        "weights": weights,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

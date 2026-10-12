#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.r2plus1d_models import dequantize_state_dict, quantize_state_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--right-weight", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def state(checkpoint):
    if "model_int8" in checkpoint:
        return dequantize_state_dict(checkpoint["model_int8"])
    return checkpoint["model"]


def main():
    args = parse_args()
    if not 0 <= args.right_weight <= 1:
        raise ValueError("Right checkpoint weight must be in [0, 1]")
    left = torch.load(args.left, map_location="cpu", weights_only=False)
    right = torch.load(args.right, map_location="cpu", weights_only=False)
    defaults = {"input_channels": 4}
    for key in ("mode", "fold", "input_channels"):
        left_value = left["config"].get(key, defaults.get(key))
        right_value = right["config"].get(key, defaults.get(key))
        if left_value != right_value:
            raise ValueError(f"Checkpoint config mismatch: {key}")
    left_state = state(left)
    right_state = state(right)
    if left_state.keys() != right_state.keys():
        raise ValueError("Checkpoint state dictionaries differ")
    mixed = {}
    for name in left_state:
        left_value = left_state[name]
        right_value = right_state[name]
        if left_value.shape != right_value.shape:
            raise ValueError(f"Tensor shape mismatch: {name}")
        if left_value.is_floating_point():
            mixed[name] = left_value.float().lerp(right_value.float(), args.right_weight).half()
        else:
            mixed[name] = right_value.clone()
    quantized, tensor_bytes = quantize_state_dict(mixed)
    config = dict(right["config"])
    config["checkpoint_interpolation"] = {
        "left": str(args.left),
        "left_sha256": hashlib.sha256(args.left.read_bytes()).hexdigest(),
        "right": str(args.right),
        "right_sha256": hashlib.sha256(args.right.read_bytes()).hexdigest(),
        "right_weight": args.right_weight,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_int8": quantized,
            "config": config,
            "epoch": right.get("epoch"),
            "metrics": {},
            "tensor_bytes": tensor_bytes,
        },
        args.output,
    )
    print(json.dumps({"output": str(args.output), "bytes": args.output.stat().st_size,
                      "tensor_bytes": tensor_bytes}, indent=2))


if __name__ == "__main__":
    main()

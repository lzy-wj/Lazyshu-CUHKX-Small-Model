#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.r2plus1d_models import quantize_state_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--right-weight", type=float, default=0.75)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 <= args.right_weight <= 1:
        raise ValueError("Right weight must be in [0, 1]")
    left = torch.load(args.left, map_location="cpu", weights_only=False)
    right = torch.load(args.right, map_location="cpu", weights_only=False)
    if int(left["config"]["fold"]) != 0 or int(right["config"]["fold"]) != 0:
        raise ValueError("This ablation is restricted to fold 0")
    left_state = {}
    for name, value in left["model"].items():
        mapped = name
        mapped = mapped.replace(".self_attn.in_proj_weight", ".attention.qkv.weight")
        mapped = mapped.replace(".self_attn.in_proj_bias", ".attention.qkv.bias")
        mapped = mapped.replace(".self_attn.out_proj.", ".attention.output.")
        mapped = mapped.replace(".norm1.", ".normalization1.")
        mapped = mapped.replace(".norm2.", ".normalization2.")
        left_state[mapped] = value
    right_state = right["model"]
    unexpected = sorted(set(left_state) - set(right_state))
    added = sorted(set(right_state) - set(left_state))
    expected_added = {
        "fusion.time_gain",
        *{
            f"fusion.layers.{layer}.attention.{name}"
            for layer in range(2)
            for name in ("relative_offset", "relative_log_scale", "modality_pair_bias")
        },
    }
    if unexpected:
        raise ValueError(f"Unmapped left tensors: {unexpected}")
    if set(added) != expected_added:
        raise ValueError(f"Unexpected right-only tensors: {added}")
    mixed = {}
    for name, right_value in right_state.items():
        if name not in left_state:
            mixed[name] = right_value.clone()
            continue
        left_value = left_state[name]
        if left_value.shape != right_value.shape:
            raise ValueError(f"Tensor shape mismatch: {name}")
        mixed[name] = (
            left_value.float().lerp(right_value.float(), args.right_weight).half()
            if left_value.is_floating_point()
            else right_value.clone()
        )
    quantized, tensor_bytes = quantize_state_dict(mixed)
    config = dict(right["config"])
    config["checkpoint_interpolation"] = {
        "left": str(args.left),
        "left_sha256": hashlib.sha256(args.left.read_bytes()).hexdigest(),
        "right": str(args.right),
        "right_sha256": hashlib.sha256(args.right.read_bytes()).hexdigest(),
        "right_weight": args.right_weight,
        "shared_tensors_interpolated": len(left_state),
        "right_only_temporal_tensors": added,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_int8": quantized,
            "config": config,
            "epoch": right["epoch"],
            "metrics": {},
            "tensor_bytes": tensor_bytes,
        },
        args.output,
    )
    print(json.dumps({"output": str(args.output), "bytes": args.output.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()

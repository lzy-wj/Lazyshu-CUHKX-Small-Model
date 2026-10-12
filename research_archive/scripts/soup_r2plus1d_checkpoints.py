#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.r2plus1d_models import quantize_state_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--weights", type=float, nargs="+")
    parser.add_argument("--storage", choices=("int8", "fp32"), default="int8")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    weights = args.weights or [1.0] * len(args.checkpoints)
    if len(weights) != len(args.checkpoints) or sum(weights) <= 0:
        raise ValueError("Checkpoint weights must match inputs and have positive sum")
    weights = [value / sum(weights) for value in weights]
    archives = [torch.load(path, map_location="cpu", weights_only=False) for path in args.checkpoints]
    configs = [archive["config"] for archive in archives]
    architecture_keys = (
        "mode", "dropout", "input_channels", "use_radar", "skeleton_representation",
        "fusion_depth", "fusion_output", "use_temporal_embedding", "use_phase_temporal_embedding",
        "visual_motion_channels", "visual_strategy", "per_visual_auxiliary",
        "visual_multiscale_tokens",
        "normalize_modality_tokens",
        "environment_auxiliary_weight",
    )
    architecture_defaults = {
        "use_radar": False,
        "use_temporal_embedding": False,
        "use_phase_temporal_embedding": False,
        "visual_motion_channels": False,
        "visual_strategy": "sequential",
        "per_visual_auxiliary": False,
        "visual_multiscale_tokens": False,
        "normalize_modality_tokens": False,
        "environment_auxiliary_weight": 0.0,
    }
    for config in configs[1:]:
        for key in architecture_keys:
            default = architecture_defaults.get(key)
            if config.get(key, default) != configs[0].get(key, default):
                raise ValueError(f"Checkpoint architecture mismatch for {key}")
    states = [archive["model"] for archive in archives]
    if any(state.keys() != states[0].keys() for state in states[1:]):
        raise ValueError("Checkpoint state dictionaries have different keys")
    soup = {}
    for name, reference in states[0].items():
        if reference.is_floating_point():
            value = sum(weight * state[name].float() for weight, state in zip(weights, states))
            soup[name] = value.to(reference.dtype)
        else:
            soup[name] = reference.clone()
    quantized, tensor_bytes = quantize_state_dict(soup)
    manifest = {
        "method": "weighted parameter soup",
        "sources": [
            {
                "path": str(path),
                "weight": weight,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path, weight in zip(args.checkpoints, weights)
        ],
        "quantized_tensor_bytes": tensor_bytes,
        "storage": args.storage,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stored_state = {"model_int8": quantized} if args.storage == "int8" else {"model": soup}
    torch.save(stored_state | {"config": configs[0], "soup": manifest}, args.output)
    report = manifest | {"output": str(args.output), "checkpoint_bytes": args.output.stat().st_size}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

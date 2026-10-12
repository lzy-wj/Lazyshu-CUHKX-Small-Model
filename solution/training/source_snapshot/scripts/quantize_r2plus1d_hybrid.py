#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.r2plus1d_models import (
    compact_quantized_state_dict,
    dequantize_state_dict,
    quantize_state_dict_int6,
    quantize_state_dict,
    quantize_state_dict_int4_grouped,
    quantize_state_dict_nf4_grouped,
)


INT8_PREFIXES = (
    "video.stem.",
    "video.layer1.",
    "video.layer2.",
    "video_head.",
    "skeleton.",
    "imu.",
    "radar.",
    "video_projection.",
    "fusion.",
    "normalization.",
    "head.",
    "skeleton_head.",
    "imu_head.",
    "radar_head.",
    "expert_gate.",
    "class_token",
    "modality_embedding",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--layer3-bits", type=int, choices=(4, 6, 8), default=4)
    parser.add_argument("--layer4-int6-blocks", type=int, nargs="*", choices=(0, 1, 2), default=())
    parser.add_argument("--deep-scheme", choices=("uniform4", "nf4"), default="uniform4")
    return parser.parse_args()


def checkpoint_state(checkpoint):
    for key in ("model_int8", "model_int7", "model_int6", "model_int4"):
        if key in checkpoint:
            return dequantize_state_dict(checkpoint[key])
    return checkpoint["model"]


def main():
    args = parse_args()
    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    state = checkpoint_state(checkpoint)
    int8, _ = quantize_state_dict(state)
    int4, _ = (
        quantize_state_dict_nf4_grouped(state, args.group_size)
        if args.deep_scheme == "nf4"
        else quantize_state_dict_int4_grouped(state, args.group_size)
    )
    use_int6 = args.layer3_bits == 6 or args.layer4_int6_blocks
    int6, _ = quantize_state_dict_int6(state) if use_int6 else (None, 0)
    hybrid = {}
    for name in state:
        if name.startswith(INT8_PREFIXES) or (
            args.layer3_bits == 8 and name.startswith("video.layer3.")
        ):
            hybrid[name] = int8[name]
        elif args.layer3_bits == 6 and name.startswith("video.layer3."):
            hybrid[name] = int6[name]
        elif any(name.startswith(f"video.layer4.{block}.") for block in args.layer4_int6_blocks):
            hybrid[name] = int6[name]
        else:
            hybrid[name] = int4[name]
    compact = compact_quantized_state_dict(hybrid)
    tensor_bytes = sum(value.numel() * value.element_size() for value in compact["buffers"].values())
    output = {
        "config": checkpoint["config"],
        "epoch": checkpoint.get("epoch"),
        "metrics": checkpoint.get("metrics", {}),
        "model_int4": compact,
        "quantization": (
            f"{args.deep_scheme} group-wise 4-bit (group size {args.group_size}); "
            f"video.layer3 uses INT{args.layer3_bits}; "
            f"video.layer4 INT6 blocks: {','.join(map(str, args.layer4_int6_blocks)) or 'none'}; "
            f"INT8 prefixes: {','.join(INT8_PREFIXES)}"
        ),
        "quantized_tensor_bytes": tensor_bytes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    report = {
        "input": str(args.input),
        "output": str(args.output),
        "checkpoint_bytes": args.output.stat().st_size,
        "quantized_tensor_bytes": tensor_bytes,
        "group_size": args.group_size,
        "layer3_bits": args.layer3_bits,
        "layer4_int6_blocks": args.layer4_int6_blocks,
        "deep_scheme": args.deep_scheme,
        "int8_prefixes": INT8_PREFIXES,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

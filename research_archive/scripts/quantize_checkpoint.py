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
    quantize_state_dict,
    quantize_state_dict_int4,
    quantize_state_dict_int4_grouped,
    quantize_state_dict_int6,
    quantize_state_dict_int7,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bits", type=int, choices=(4, 6, 7, 8), default=8)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--int8-prefix", nargs="*", default=[])
    parser.add_argument("--int4-group-size", type=int, default=0)
    args = parser.parse_args()
    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    state = (
        checkpoint["model"] if "model" in checkpoint else
        dequantize_state_dict(checkpoint["model_int8"]) if "model_int8" in checkpoint else
        dequantize_state_dict(checkpoint["model_int7"]) if "model_int7" in checkpoint else
        dequantize_state_dict(checkpoint["model_int6"]) if "model_int6" in checkpoint else
        dequantize_state_dict(checkpoint["model_int4"])
    )
    model_key = f"model_int{args.bits}"
    if args.int8_prefix and args.bits == 8:
        raise ValueError("Mixed precision requires a sub-8-bit base")
    if args.int4_group_size and args.bits != 4:
        raise ValueError("Grouped INT4 requires --bits 4")
    if args.compact and model_key in checkpoint and not args.int8_prefix:
        quantized = checkpoint[model_key]
        tensor_bytes = checkpoint.get("quantized_tensor_bytes", checkpoint.get("tensor_bytes", 0))
    else:
        low_precision = {name: value for name, value in state.items()
                         if not any(name.startswith(prefix) for prefix in args.int8_prefix)}
        high_precision = {name: value for name, value in state.items()
                          if any(name.startswith(prefix) for prefix in args.int8_prefix)}
        quantized, tensor_bytes = (
            quantize_state_dict_int4_grouped(low_precision, args.int4_group_size)
            if args.bits == 4 and args.int4_group_size else quantize_state_dict_int4(low_precision)
            if args.bits == 4 else quantize_state_dict_int6(low_precision)
            if args.bits == 6 else quantize_state_dict_int7(low_precision)
            if args.bits == 7 else quantize_state_dict(low_precision)
        )
        if high_precision:
            promoted, promoted_bytes = quantize_state_dict(high_precision)
            quantized.update(promoted)
            tensor_bytes += promoted_bytes
    output = {key: value for key, value in checkpoint.items() if not key.startswith("model")}
    already_compact = isinstance(quantized, dict) and quantized.get("format") == "compact_quantized_state_dict_v1"
    output[model_key] = compact_quantized_state_dict(quantized) if args.compact and not already_compact else quantized
    output["quantization"] = (
        f"symmetric group-wise INT4 (group size {args.int4_group_size}); remaining floating tensors FP16"
        if args.int4_group_size else
        f"symmetric per-output-channel INT{args.bits}; remaining floating tensors FP16"
    )
    if args.int8_prefix:
        output["quantization"] += f"; INT8 prefixes: {','.join(args.int8_prefix)}"
    output["quantized_tensor_bytes"] = tensor_bytes
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(
        json.dumps(
            {
                "input": str(args.input),
                "input_bytes": args.input.stat().st_size,
                "output": str(args.output),
                "output_bytes": args.output.stat().st_size,
                "tensor_bytes": tensor_bytes,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

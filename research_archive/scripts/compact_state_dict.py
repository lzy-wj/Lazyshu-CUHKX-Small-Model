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
    expand_compact_quantized_state_dict,
    quantize_state_dict,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bits", choices=("exact", "8"), default="exact")
    args = parser.parse_args()
    state = torch.load(args.input, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or not state or not all(torch.is_tensor(value) for value in state.values()):
        raise ValueError("Input must be a plain tensor state dict")
    if args.bits == "8":
        quantized, tensor_bytes = quantize_state_dict(state)
        compact = compact_quantized_state_dict(quantized)
        restored = dequantize_state_dict(compact)
    else:
        tensor_bytes = sum(value.numel() * value.element_size() for value in state.values())
        compact = compact_quantized_state_dict(state)
        restored = expand_compact_quantized_state_dict(compact)
        if state.keys() != restored.keys() or not all(torch.equal(state[name], restored[name]) for name in state):
            raise RuntimeError("Compact state dict failed exact round-trip validation")
    if state.keys() != restored.keys() or any(state[name].shape != restored[name].shape for name in state):
        raise RuntimeError("Compact state dict failed structural validation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    key = "model_int8" if args.bits == "8" else "state_dict_compact"
    torch.save({key: compact, "tensor_bytes": tensor_bytes}, args.output)
    print(json.dumps({
        "input": str(args.input),
        "input_bytes": args.input.stat().st_size,
        "output": str(args.output),
        "output_bytes": args.output.stat().st_size,
        "precision": args.bits,
    }, indent=2))


if __name__ == "__main__":
    main()

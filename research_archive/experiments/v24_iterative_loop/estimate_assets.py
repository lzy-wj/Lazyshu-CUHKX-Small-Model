#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from iterative_model import build_from_source
from src.r2plus1d_models import quantize_state_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "experiments/scene_context_v13/outputs/v23_parameter_soup_warm60_fp32_f0.pt",
    )
    parser.add_argument("--output", type=Path, default=HERE / "asset_estimate.json")
    return parser.parse_args()


def main():
    args = parse_args()
    model, _, _ = build_from_source(args.checkpoint, iterations=1)
    state = model.state_dict()
    _, int8_tensor_bytes = quantize_state_dict(state)
    tokens = 45
    dimension = 192
    feedforward = 576
    fusion_layers = 2
    fusion_flops = fusion_layers * (
        4 * tokens * dimension ** 2
        + 2 * tokens ** 2 * dimension
        + 2 * tokens * dimension * feedforward
    )
    base_parameters = sum(parameter.numel() for parameter in model.base.parameters())
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    extra_state_bytes_fp16 = sum(
        value.numel() * (2 if value.is_floating_point() else value.element_size())
        for name, value in state.items()
        if not name.startswith("base.")
    )
    report = {
        "source_checkpoint": str(args.checkpoint),
        "base_parameters": base_parameters,
        "iterative_parameters": total_parameters,
        "incremental_parameters": total_parameters - base_parameters,
        "parameter_count_is_independent_of_k": True,
        "extra_state_bytes_fp16": extra_state_bytes_fp16,
        "estimated_full_state_bytes_fp16": sum(
            value.numel() * (2 if value.is_floating_point() else value.element_size())
            for value in state.values()
        ),
        "exact_int8_tensor_bytes": int8_tensor_bytes,
        "deployment_target_bytes": 95_000_000,
        "under_target": int8_tensor_bytes < 95_000_000,
        "cached_token_bytes_per_sample_bf16_v23": 44 * 192 * 2,
        "fusion_flops_per_iteration_per_sample_estimate": fusion_flops,
        "assumptions": "V23 has 3x4 visual + 16 skeleton + 16 IMU tokens; two fusion layers, d=192, FFN=576.",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

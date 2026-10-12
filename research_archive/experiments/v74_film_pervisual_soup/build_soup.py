#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.r2plus1d_models import quantize_state_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--v65-weight", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 <= args.v65_weight <= 1:
        raise ValueError("V65 weight must be in [0, 1]")
    paths = [
        ROOT
        / f"experiments/v55_modality_film/outputs/checkpoints/v55_modality_film_f{args.fold}/best.pt",
        ROOT
        / f"experiments/v65_film_pervisual_aux/outputs/checkpoints/v65_film_pervisual_aux_f{args.fold}/best.pt",
    ]
    archives = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    states = [archive["model"] for archive in archives]
    if states[0].keys() != states[1].keys():
        raise ValueError("V55 and V65 state dictionaries differ")
    weights = (1 - args.v65_weight, args.v65_weight)
    soup = {}
    for name, reference in states[0].items():
        if states[1][name].shape != reference.shape:
            raise ValueError(f"State shape mismatch: {name}")
        if reference.is_floating_point():
            soup[name] = (
                weights[0] * reference.float() + weights[1] * states[1][name].float()
            ).to(reference.dtype)
        else:
            soup[name] = reference.clone()
    quantized, tensor_bytes = quantize_state_dict(soup)
    config = archives[0]["config"].copy()
    config["per_visual_auxiliary"] = False
    manifest = {
        "method": "V55/V65 weighted parameter soup",
        "fold": args.fold,
        "sources": [
            {
                "path": str(path),
                "weight": weight,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path, weight in zip(paths, weights)
        ],
        "quantized_tensor_bytes": tensor_bytes,
        "inference_equivalence": (
            "per_visual_auxiliary changes only the training loss and returned auxiliary dictionary"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_int8": quantized, "config": config, "soup": manifest}, args.output)
    report = manifest | {
        "output": str(args.output),
        "checkpoint_bytes": args.output.stat().st_size,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

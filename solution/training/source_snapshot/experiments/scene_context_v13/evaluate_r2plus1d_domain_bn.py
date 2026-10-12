#!/usr/bin/env python3
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_r2plus1d_checkpoint import adapt_batch_norm, build_model, make_loader, predict
from src.data import get_indices, get_records
from src.r2plus1d_models import dequantize_state_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument(
        "--adaptation-set",
        choices=("target_env", "source_env", "source_target_env"),
        default="target_env",
    )
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--tta-hflip", action="store_true")
    parser.add_argument("--tta-crop-scale", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def checkpoint_state(checkpoint):
    for key in ("model_int4", "model_int7", "model_int6", "model_int8"):
        if key in checkpoint:
            return dequantize_state_dict(checkpoint[key])
    return checkpoint["model"]


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    mode = config["mode"]
    state = checkpoint_state(checkpoint)
    records = get_records()
    train_indices, valid_indices = get_indices(records, args.fold)
    rows = []
    groups = []
    for environment in (0, 1):
        target = [
            index for index in valid_indices
            if int(records[index]["user"] > 15) == environment
        ]
        source = [
            index for index in train_indices
            if int(records[index]["user"] > 15) == environment
        ]
        adaptation = {
            "target_env": target,
            "source_env": source,
            "source_target_env": source + target,
        }[args.adaptation_set]
        model = build_model(config, copy.deepcopy(state))
        adaptation_loader = make_loader(adaptation, args, config, mode)
        adapt_batch_norm(model, adaptation_loader, mode, args.passes, reset=True)
        target_loader = make_loader(target, args, config, mode)
        rows.extend(predict(model, target_loader, mode, args.tta_hflip, args.tta_crop_scale))
        groups.append(
            {
                "environment": environment,
                "adaptation_rows": len(adaptation),
                "target_rows": len(target),
            }
        )
        del model
        torch.cuda.empty_cache()
    ordered = {index: (label, values) for index, label, values, _ in rows}
    indices = np.asarray(valid_indices, dtype=np.int64)
    labels = np.asarray([ordered[int(index)][0] for index in indices])
    probabilities = np.stack([ordered[int(index)][1] for index in indices])
    report = {
        "accuracy": float((probabilities.argmax(1) == labels).mean()),
        "fold": args.fold,
        "adaptation_set": args.adaptation_set,
        "passes": args.passes,
        "groups": groups,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, probabilities=probabilities, labels=labels, indices=indices)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

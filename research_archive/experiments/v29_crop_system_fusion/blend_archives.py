#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapted", type=Path, required=True)
    parser.add_argument("--crop", type=Path, required=True)
    parser.add_argument("--adapted-weight", type=float, default=0.4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 <= args.adapted_weight <= 1:
        raise ValueError("adapted weight must be in [0, 1]")
    adapted = np.load(args.adapted)
    crop = np.load(args.crop)
    if not np.array_equal(adapted["indices"], crop["indices"]):
        raise ValueError("prediction indices are not aligned")
    if "labels" in adapted and "labels" in crop:
        if not np.array_equal(adapted["labels"], crop["labels"]):
            raise ValueError("validation labels are not aligned")
    probabilities = (
        args.adapted_weight * adapted["probabilities"]
        + (1 - args.adapted_weight) * crop["probabilities"]
    ).astype(np.float32)
    payload = {
        "probabilities": probabilities,
        "indices": adapted["indices"],
    }
    if "labels" in adapted:
        payload["labels"] = adapted["labels"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **payload)
    report = {
        "adapted": str(args.adapted),
        "crop": str(args.crop),
        "adapted_weight": args.adapted_weight,
        "crop_weight": 1 - args.adapted_weight,
        "rows": len(probabilities),
        "classes_predicted": int(np.unique(probabilities.argmax(axis=1)).size),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

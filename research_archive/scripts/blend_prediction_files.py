#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--weights", type=float, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    weights = np.asarray(args.weights or [1] * len(args.inputs), dtype=np.float64)
    if len(weights) != len(args.inputs) or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Supply one nonnegative weight per input, with positive total")
    weights /= weights.sum()
    values = [np.load(path) for path in args.inputs]
    indices = values[0]["indices"]
    if any(not np.array_equal(indices, item["indices"]) for item in values[1:]):
        raise ValueError("Prediction indices do not align")
    stacked = np.stack([item["probabilities"] for item in values])
    probabilities = np.sum(
        stacked * weights.astype(stacked.dtype)[:, None, None], axis=0, dtype=stacked.dtype
    )
    payload = {"probabilities": probabilities, "indices": indices}
    if "labels" in values[0]:
        labels = values[0]["labels"]
        if any("labels" not in item or not np.array_equal(labels, item["labels"]) for item in values[1:]):
            raise ValueError("Prediction labels do not align")
        payload["labels"] = labels
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **payload)
    print(json.dumps({
        "output": str(args.output),
        "inputs": [str(path) for path in args.inputs],
        "weights": weights.tolist(),
        "rows": len(indices),
    }, indent=2))


if __name__ == "__main__":
    main()

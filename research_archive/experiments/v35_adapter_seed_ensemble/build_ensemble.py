#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    archives = [np.load(path) for path in args.inputs]
    reference = archives[0]
    for archive in archives[1:]:
        if not np.array_equal(reference["indices"], archive["indices"]):
            raise ValueError("adapter predictions are not aligned")
        if not np.array_equal(reference["labels"], archive["labels"]):
            raise ValueError("adapter labels are not aligned")
    probabilities = np.mean(
        np.stack([archive["probabilities"] for archive in archives]), axis=0
    ).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        probabilities=probabilities,
        indices=reference["indices"],
        labels=reference["labels"],
    )
    print(json.dumps({"inputs": [str(path) for path in args.inputs], "rows": len(probabilities)}, indent=2))


if __name__ == "__main__":
    main()

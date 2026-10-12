#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fallback", type=Path, required=True)
    parser.add_argument("--specialized", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    fallback = np.load(args.fallback)
    indices = fallback["indices"]
    labels = fallback["labels"]
    probabilities = fallback["probabilities"].copy()
    positions = {int(index): position for position, index in enumerate(indices)}
    replaced = 0
    sources = []
    for path in args.specialized:
        archive = np.load(path)
        for index, values in zip(archive["indices"], archive["probabilities"]):
            probabilities[positions[int(index)]] = values
            replaced += 1
        sources.append(str(path))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, probabilities=probabilities, labels=labels, indices=indices)
    report = {
        "fallback": str(args.fallback),
        "specialized": sources,
        "replaced": replaced,
        "accuracy": float((probabilities.argmax(1) == labels).mean()),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lowres", type=Path, required=True)
    parser.add_argument("--highres", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def load(path):
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def main():
    args = parse_args()
    lowres = load(args.lowres)
    highres = load(args.highres)
    positions = {int(index): position for position, index in enumerate(highres["indices"])}
    order = np.asarray([positions[int(index)] for index in lowres["indices"]])
    if len(order) != 405 or len(set(order.tolist())) != 405:
        raise ValueError("Expected one-to-one coverage of all 405 test rows")
    highres_probabilities = highres["probabilities"][order]
    probabilities = 0.5 * (
        lowres["probabilities"].astype(np.float64)
        + highres_probabilities.astype(np.float64)
    )
    report = {
        "protocol": "Equal arithmetic mean of pseudo-adapted 128px and 160px probabilities.",
        "rows": len(probabilities),
        "raw_disagreement": int(
            (
                lowres["probabilities"].argmax(axis=1)
                != highres_probabilities.argmax(axis=1)
            ).sum()
        ),
        "mean_confidence": float(probabilities.max(axis=1).mean()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        probabilities=probabilities.astype(np.float32),
        indices=lowres["indices"],
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

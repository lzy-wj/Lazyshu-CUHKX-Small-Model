#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-fold0", type=Path, required=True)
    parser.add_argument("--base-fold1", type=Path, required=True)
    parser.add_argument("--candidate-fold0", type=Path, required=True)
    parser.add_argument("--candidate-fold1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--availability", type=Path, default=Path("data/processed/r2plus1d_yolo_16_128/available.npy"))
    parser.add_argument("--candidate-weight", type=float)
    parser.add_argument("--candidate-temperature", type=float)
    return parser.parse_args()


def load(path):
    values = np.load(path)
    return {key: values[key] for key in values.files}


def align(base, candidate):
    positions = {int(index): position for position, index in enumerate(candidate["indices"])}
    order = np.asarray([positions[int(index)] for index in base["indices"]])
    if "labels" in candidate and not np.array_equal(base["labels"], candidate["labels"][order]):
        raise ValueError("Candidate labels do not align with base labels")
    return candidate["probabilities"][order]


def temperature_scale(probabilities, temperature):
    values = np.log(np.clip(probabilities, 1e-12, 1)) / temperature
    values -= values.max(axis=1, keepdims=True)
    values = np.exp(values)
    return values / values.sum(axis=1, keepdims=True)


def accuracy(labels, probabilities):
    return float((probabilities.argmax(axis=1) == labels).mean())


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base = [load(args.base_fold0), load(args.base_fold1)]
    candidates = [load(args.candidate_fold0), load(args.candidate_fold1)]
    candidate_probabilities = [align(left, right) for left, right in zip(base, candidates)]
    if (args.candidate_weight is None) != (args.candidate_temperature is None):
        raise ValueError("Supply both fixed candidate parameters or neither")
    if args.candidate_weight is not None:
        if not 0 <= args.candidate_weight <= 1 or args.candidate_temperature <= 0:
            raise ValueError("Invalid fixed candidate parameters")
        selected = (None, args.candidate_weight, args.candidate_temperature)
    else:
        grid = []
        for weight in np.linspace(0, 0.8, 17):
            for temperature in (0.6, 0.8, 1.0, 1.2, 1.5, 2.0):
                probability = (
                    (1 - weight) * base[0]["probabilities"]
                    + weight * temperature_scale(candidate_probabilities[0], temperature)
                )
                grid.append((accuracy(base[0]["labels"], probability), weight, temperature))
        selected = max(grid, key=lambda values: (values[0], -values[1]))
    availability = np.load(args.availability).astype(bool)
    report = {
        "selection_fold": 0,
        "selection": "fixed" if args.candidate_weight is not None else "fold0_grid",
        "candidate_weight": selected[1],
        "candidate_temperature": selected[2],
        "folds": [],
    }
    for fold in (0, 1):
        probability = (
            (1 - selected[1]) * base[fold]["probabilities"]
            + selected[1] * temperature_scale(candidate_probabilities[fold], selected[2])
        )
        labels = base[fold]["labels"]
        indices = base[fold]["indices"]
        valid = availability[indices]
        report["folds"].append(
            {
                "fold": fold,
                "base_accuracy": accuracy(labels, base[fold]["probabilities"]),
                "candidate_accuracy": accuracy(labels, candidate_probabilities[fold]),
                "candidate_available_accuracy": accuracy(labels[valid], candidate_probabilities[fold][valid]),
                "candidate_coverage": float(valid.mean()),
                "blend_accuracy": accuracy(labels, probability),
            }
        )
        np.savez(
            args.output / f"blend_fold{fold}.npz",
            probabilities=probability,
            labels=labels,
            indices=indices,
        )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

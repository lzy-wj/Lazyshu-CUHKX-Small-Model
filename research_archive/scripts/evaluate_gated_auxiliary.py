#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_repeat_decoder import decode, transition_matrix
from src.data import FOLDS, get_records


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-fold0", type=Path, required=True)
    parser.add_argument("--base-fold1", type=Path, required=True)
    parser.add_argument("--aux-fold0", type=Path, required=True)
    parser.add_argument("--aux-fold1", type=Path, required=True)
    parser.add_argument("--maximum-confidence", type=float, default=0.15)
    parser.add_argument("--auxiliary-weight", type=float, default=0.1)
    parser.add_argument("--auxiliary-temperature", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def temperature_scale(probabilities, temperature):
    values = np.log(np.clip(probabilities, 1e-12, 1)) / temperature
    values -= values.max(axis=1, keepdims=True)
    values = np.exp(values)
    return values / values.sum(axis=1, keepdims=True)


def align(base, auxiliary):
    positions = {int(index): position for position, index in enumerate(auxiliary["indices"])}
    order = np.asarray([positions[int(index)] for index in base["indices"]])
    if not np.array_equal(base["labels"], auxiliary["labels"][order]):
        raise ValueError("Prediction labels do not align")
    return auxiliary["probabilities"][order]


def main():
    args = parse_args()
    if not 0 <= args.maximum_confidence <= 1 or not 0 <= args.auxiliary_weight <= 1:
        raise ValueError("Invalid gating parameters")
    if args.auxiliary_temperature <= 0:
        raise ValueError("Auxiliary temperature must be positive")
    records = get_records()
    base_paths = [args.base_fold0, args.base_fold1]
    auxiliary_paths = [args.aux_fold0, args.aux_fold1]
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "maximum_confidence": args.maximum_confidence,
        "auxiliary_weight": args.auxiliary_weight,
        "auxiliary_temperature": args.auxiliary_temperature,
        "repeat_pool_power": 2,
        "transition_strength": 0.4,
        "folds": [],
    }
    for fold in (0, 1):
        base = np.load(base_paths[fold])
        auxiliary = align(base, np.load(auxiliary_paths[fold]))
        auxiliary = temperature_scale(auxiliary, args.auxiliary_temperature)
        probabilities = base["probabilities"].copy()
        selected = probabilities.max(axis=1) < args.maximum_confidence
        probabilities[selected] = (
            (1 - args.auxiliary_weight) * probabilities[selected]
            + args.auxiliary_weight * auxiliary[selected]
        )
        transitions = [
            transition_matrix(records, set(FOLDS[fold]), environment=environment)
            for environment in (0, 1)
        ]
        decoded = decode(
            probabilities,
            base["indices"],
            records,
            maximum_gap=120,
            minimum_similarity=0.7,
            enforce_unique=True,
            transition_strength=0.4,
            environment_transitions=transitions,
            repeat_pool_power=2,
        )
        raw_accuracy = float((probabilities.argmax(1) == base["labels"]).mean())
        report["folds"].append(
            {
                "fold": fold,
                "gated_rows": int(selected.sum()),
                "raw_accuracy": raw_accuracy,
                "decoded_accuracy": decoded["accuracy"],
            }
        )
        np.savez(
            args.output / f"fold{fold}.npz",
            probabilities=probabilities,
            labels=base["labels"],
            indices=base["indices"],
        )
    total_correct = sum(
        round(item["decoded_accuracy"] * count)
        for item, count in zip(report["folds"], (652, 705))
    )
    report["decoded_weighted_accuracy"] = total_correct / 1357
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

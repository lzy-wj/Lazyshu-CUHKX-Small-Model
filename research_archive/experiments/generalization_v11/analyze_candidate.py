#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data import get_records


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--ensemble", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path):
    values = np.load(path)
    return {key: values[key] for key in values.files}


def align(reference, candidate):
    positions = {int(index): position for position, index in enumerate(candidate["indices"])}
    order = np.asarray([positions[int(index)] for index in reference["indices"]])
    if not np.array_equal(reference["labels"], candidate["labels"][order]):
        raise ValueError("Labels do not align")
    return candidate["probabilities"][order]


def grouped(probabilities, labels, indices, records, branch_presence):
    predictions = probabilities.argmax(1)
    users = np.asarray([records[int(index)]["user"] for index in indices])
    per_user = {
        str(user): float((predictions[users == user] == labels[users == user]).mean())
        for user in np.unique(users)
    }
    values = np.asarray(list(per_user.values()))
    class_accuracies = [
        (predictions[labels == label] == labels[labels == label]).mean()
        for label in np.unique(labels)
    ]
    confidence = probabilities.max(1)
    correctness = predictions == labels
    calibration_error = 0.0
    for lower in np.linspace(0, 0.9, 10):
        selected = (confidence >= lower) & (confidence < lower + 0.1)
        if selected.any():
            calibration_error += selected.mean() * abs(correctness[selected].mean() - confidence[selected].mean())
    patterns = {}
    encoded = branch_presence @ np.asarray([4, 2, 1])
    for pattern in np.unique(encoded):
        selected = encoded == pattern
        name = format(int(pattern), "03b")
        patterns[name] = {
            "clips": int(selected.sum()),
            "accuracy": float(correctness[selected].mean()),
        }
    return {
        "accuracy": float((predictions == labels).mean()),
        "macro_class_accuracy": float(np.mean(class_accuracies)),
        "user_macro_accuracy": float(values.mean()),
        "worst_user_accuracy": float(values.min()),
        "user_accuracy_std": float(values.std()),
        "nll": float(-np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1)).mean()),
        "ece_10_bin": float(calibration_error),
        "per_user_accuracy": per_user,
        "branch_presence_pattern": patterns,
    }


def main():
    args = parse_args()
    records = get_records()
    baseline_standalone = load(
        ROOT / f"outputs/study_r2plus1d/r2_staged_soup_f{args.fold}_w075_crop075.npz"
    )
    baseline_ensemble = load(ROOT / f"experiments/baseline_fold{args.fold}.npz")
    candidate = load(args.candidate)
    ensemble = load(args.ensemble)
    labels = baseline_ensemble["labels"]
    indices = baseline_ensemble["indices"]
    visual = np.load(ROOT / "data/processed/r2plus1d_yolo_32_128/available.npy").astype(bool)[indices]
    sensors = np.load(ROOT / "data/processed/sensor_mask.npy").astype(bool)[indices, :2]
    branch_presence = np.concatenate([visual[:, None], sensors], axis=1)
    report = {
        "fold": args.fold,
        "standalone": {
            "baseline": grouped(
                align(baseline_ensemble, baseline_standalone), labels, indices, records, branch_presence
            ),
            "candidate": grouped(
                align(baseline_ensemble, candidate), labels, indices, records, branch_presence
            ),
        },
        "ensemble": {
            "baseline": grouped(
                baseline_ensemble["probabilities"], labels, indices, records, branch_presence
            ),
            "candidate": grouped(
                align(baseline_ensemble, ensemble), labels, indices, records, branch_presence
            ),
        },
    }
    for section in ("standalone", "ensemble"):
        report[section]["delta"] = {
            key: report[section]["candidate"][key] - report[section]["baseline"][key]
            for key in (
                "accuracy", "macro_class_accuracy", "user_macro_accuracy", "worst_user_accuracy",
                "user_accuracy_std", "nll", "ece_10_bin",
            )
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

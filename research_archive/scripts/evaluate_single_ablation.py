#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_repeat_decoder import decode, transition_matrix
from src.data import FOLDS, ROOT, get_records


BASELINE = {
    0: {
        "standalone_accuracy": 0.6779141104294478,
        "available_accuracy": 0.6949685534591195,
        "ensemble_raw_accuracy": 0.7101226993865031,
        "decoded_accuracy": 0.7684049079754601,
    },
    1: {
        "standalone_accuracy": 0.7191489361702128,
        "available_accuracy": 0.7897196261682243,
        "ensemble_raw_accuracy": 0.7914893617021277,
        "decoded_accuracy": 0.8425531914893617,
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--availability", type=Path,
                        default=ROOT / "data/processed/r2plus1d_yolo_32_128/available.npy")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path):
    values = np.load(path)
    return {key: values[key] for key in values.files}


def align(reference, candidate):
    positions = {int(index): position for position, index in enumerate(candidate["indices"])}
    order = np.asarray([positions[int(index)] for index in reference["indices"]])
    if "labels" in candidate and not np.array_equal(reference["labels"], candidate["labels"][order]):
        raise ValueError("Prediction labels do not align")
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
    base = load(ROOT / f"outputs/study_r2plus1d/lightweight_base_int8/base_fold{args.fold}.npz")
    candidate_file = load(args.candidate)
    candidate = align(base, candidate_file)
    auxiliary = align(
        base,
        load(ROOT / f"outputs/study_x3dl_soup/selected_cropblend50_fold{args.fold}.npz"),
    )
    probabilities = 0.4 * base["probabilities"] + 0.6 * temperature_scale(candidate, 2.0)
    gated = probabilities.max(axis=1) < 0.15
    probabilities[gated] = 0.9 * probabilities[gated] + 0.1 * auxiliary[gated]
    records = get_records()
    transitions = [
        transition_matrix(records, set(FOLDS[args.fold]), environment=environment)
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
    available = np.load(args.availability).astype(bool)[base["indices"]]
    metrics = {
        "fold": args.fold,
        "candidate": str(args.candidate),
        "candidate_sha256": hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
        "rows": len(base["indices"]),
        "gated_rows": int(gated.sum()),
        "standalone_accuracy": accuracy(base["labels"], candidate),
        "available_accuracy": accuracy(base["labels"][available], candidate[available]),
        "coverage": float(available.mean()),
        "ensemble_raw_accuracy": accuracy(base["labels"], probabilities),
        "decoded_accuracy": decoded["accuracy"],
        "decoded_correct": round(decoded["accuracy"] * len(base["indices"])),
        "baseline": BASELINE[args.fold],
    }
    metrics["delta_percentage_points"] = {
        key: 100 * (metrics[key] - BASELINE[args.fold][key])
        for key in ("standalone_accuracy", "available_accuracy", "ensemble_raw_accuracy", "decoded_accuracy")
    }
    standalone_delta = metrics["delta_percentage_points"]["standalone_accuracy"]
    raw_delta = metrics["delta_percentage_points"]["ensemble_raw_accuracy"]
    decoded_delta = metrics["decoded_correct"] - round(
        BASELINE[args.fold]["decoded_accuracy"] * len(base["indices"])
    )
    metrics["acceptance_gate"] = {
        "standalone_gain_at_least_0_5pp_without_decoded_regression": (
            standalone_delta >= 0.5 and decoded_delta >= 0
        ),
        "decoded_gain_at_least_2_without_raw_regression": decoded_delta >= 2 and raw_delta >= 0,
    }
    metrics["advance_to_fold1"] = any(metrics["acceptance_gate"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez(
        args.output.with_suffix(".npz"),
        probabilities=probabilities,
        labels=base["labels"],
        indices=base["indices"],
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

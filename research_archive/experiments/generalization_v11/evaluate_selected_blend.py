#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_repeat_decoder import decode, transition_matrix
from scripts.evaluate_single_ablation import BASELINE, accuracy, align, load, temperature_scale
from src.data import FOLDS, get_records


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("selected_config.json"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    config = json.loads(args.config.read_text())
    base = load(ROOT / f"outputs/study_r2plus1d/lightweight_base_int8/base_fold{args.fold}.npz")
    candidate_file = load(args.candidate)
    candidate = align(base, candidate_file)
    auxiliary = align(
        base,
        load(ROOT / f"outputs/study_x3dl_soup/selected_cropblend50_fold{args.fold}.npz"),
    )
    r2_weight = float(config["r2plus1d_weight"])
    probabilities = (1 - r2_weight) * base["probabilities"] + r2_weight * temperature_scale(
        candidate, float(config["r2plus1d_temperature"])
    )
    gated = probabilities.max(axis=1) < float(config["x3d_l_maximum_base_confidence"])
    auxiliary_weight = float(config["x3d_l_auxiliary_weight"])
    probabilities[gated] = (
        (1 - auxiliary_weight) * probabilities[gated] + auxiliary_weight * auxiliary[gated]
    )
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
        transition_strength=float(config["transition_strength"]),
        environment_transitions=transitions,
        repeat_pool_power=float(config["repeat_pool_power"]),
    )
    metrics = {
        "fold": args.fold,
        "candidate": str(args.candidate),
        "candidate_sha256": hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
        "config": str(args.config),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "rows": len(base["indices"]),
        "gated_rows": int(gated.sum()),
        "standalone_accuracy": accuracy(base["labels"], candidate),
        "ensemble_raw_accuracy": accuracy(base["labels"], probabilities),
        "decoded_accuracy": float(decoded["accuracy"]),
        "decoded_correct": round(float(decoded["accuracy"]) * len(base["indices"])),
        "baseline": BASELINE[args.fold],
    }
    metrics["delta_percentage_points"] = {
        key: 100 * (metrics[key] - BASELINE[args.fold][key])
        for key in ("standalone_accuracy", "ensemble_raw_accuracy", "decoded_accuracy")
    }
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

#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.research_sequence_decoder import selected_probabilities
from scripts.evaluate_repeat_decoder import decode, transition_matrix
from src.data import FOLDS, get_records


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/scene_context_v13/outputs/decoder_grid.json",
    )
    return parser.parse_args()


def evaluate(values, records, transitions, settings):
    probabilities, indices, labels = values
    result = decode(
        probabilities,
        indices,
        records,
        maximum_gap=settings["maximum_gap"],
        minimum_similarity=settings["minimum_similarity"],
        enforce_unique=True,
        transition_strength=settings["transition_strength"],
        environment_transitions=transitions,
        repeat_pool_power=settings["repeat_pool_power"],
        beam_size=settings["beam_size"],
    )
    return {
        "correct": round(result["accuracy"] * len(labels)),
        "accuracy": result["accuracy"],
        "groups": result["groups"],
        "grouped_records": result["grouped_records"],
    }


def main():
    args = parse_args()
    records = get_records()
    folds = [selected_probabilities(fold) for fold in (0, 1)]
    transitions = [
        [
            transition_matrix(records, set(FOLDS[fold]), environment=environment)
            for environment in (0, 1)
        ]
        for fold in (0, 1)
    ]
    candidates = []
    for maximum_gap in (60, 90, 120, 180, 240):
        for minimum_similarity in (0.55, 0.65, 0.7, 0.75, 0.8):
            for repeat_pool_power in (1.0, 1.5, 2.0, 3.0):
                for transition_strength in (0.2, 0.3, 0.4, 0.5, 0.6):
                    settings = {
                        "maximum_gap": maximum_gap,
                        "minimum_similarity": minimum_similarity,
                        "repeat_pool_power": repeat_pool_power,
                        "transition_strength": transition_strength,
                        "beam_size": 100,
                    }
                    candidates.append(
                        settings | {"fold0": evaluate(folds[0], records, transitions[0], settings)}
                    )
    selected = max(
        candidates,
        key=lambda item: (
            item["fold0"]["correct"],
            -abs(item["transition_strength"] - 0.4),
            -abs(item["repeat_pool_power"] - 1.5),
            -abs(item["minimum_similarity"] - 0.7),
            -abs(item["maximum_gap"] - 120),
        ),
    )
    settings = {key: selected[key] for key in (
        "maximum_gap", "minimum_similarity", "repeat_pool_power", "transition_strength", "beam_size"
    )}
    selected["fold1"] = evaluate(folds[1], records, transitions[1], settings)
    combined_correct = selected["fold0"]["correct"] + selected["fold1"]["correct"]
    report = {
        "protocol": "Decoder grid selected on fold 0 only; fold 1 evaluated once after selection.",
        "selected": selected,
        "combined": {
            "correct": combined_correct,
            "rows": len(folds[0][2]) + len(folds[1][2]),
            "accuracy": combined_correct / (len(folds[0][2]) + len(folds[1][2])),
        },
        "baseline": {"correct": 1103, "accuracy": 1103 / 1357},
        "top_fold0": sorted(
            candidates, key=lambda item: item["fold0"]["correct"], reverse=True
        )[:25],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "top_fold0"}, indent=2))


if __name__ == "__main__":
    main()

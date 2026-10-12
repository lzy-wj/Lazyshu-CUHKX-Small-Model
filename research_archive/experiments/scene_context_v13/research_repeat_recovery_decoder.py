#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.research_flexible_pair_decoder import evaluate, prepare_fold
from experiments.scene_context_v13.research_paired_script_decoder import PAIR_SOURCES
from experiments.scene_context_v13.research_sequence_decoder import selected_probabilities
from scripts.evaluate_repeat_decoder import pool_repeat_probabilities, repeat_groups, rounds
from src.data import FOLDS, get_records


PAIRED_USERS = ({8, 18}, {7, 17})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/scene_context_v13/outputs/repeat_recovery_decoder.json",
    )
    return parser.parse_args()


def prepool(probabilities, indices, records, maximum_gap, minimum_similarity, power):
    probability_by_index = {
        int(index): probability.copy() for index, probability in zip(indices, probabilities)
    }
    round_values = rounds(indices, records)
    groups = repeat_groups(round_values, probability_by_index, maximum_gap, minimum_similarity)
    for group in groups:
        pooled = pool_repeat_probabilities(
            np.stack(
                [[probability_by_index[index] for index in item["rows"]] for item in group]
            ),
            power,
        )
        for item in group:
            for position, index in enumerate(item["rows"]):
                probability_by_index[index] = pooled[position]
    return np.stack([probability_by_index[int(index)] for index in indices]), len(groups)


def run(fold, fold_values, records, settings):
    probabilities, indices, labels = fold_values
    if settings["enabled"]:
        probabilities, recovered_groups = prepool(
            probabilities,
            indices,
            records,
            settings["maximum_gap"],
            settings["minimum_similarity"],
            settings["power"],
        )
    else:
        recovered_groups = 0
    prepared = prepare_fold(
        probabilities,
        indices,
        labels,
        records,
        PAIR_SOURCES[fold],
        FOLDS[fold],
    )
    result = evaluate(
        prepared,
        template_strength=1.8,
        gap_penalty=0.8,
        maximum_edits=5,
        allowed_users=PAIRED_USERS[fold],
        alignment_beam_size=40,
    )
    result["prepool_groups"] = recovered_groups
    return result


def main():
    args = parse_args()
    records = get_records()
    folds = [selected_probabilities(fold) for fold in (0, 1)]
    settings_grid = [{"enabled": False, "maximum_gap": 120, "minimum_similarity": 0.7, "power": 1.5}]
    for maximum_gap in (120, 180, 240):
        for minimum_similarity in (0.4, 0.5, 0.6, 0.7):
            for power in (1.0, 1.5):
                settings_grid.append(
                    {
                        "enabled": True,
                        "maximum_gap": maximum_gap,
                        "minimum_similarity": minimum_similarity,
                        "power": power,
                    }
                )
    candidates = []
    for settings in settings_grid:
        result = run(0, folds[0], records, settings)
        candidates.append(settings | {"fold0": result})
        print(settings, result["correct"], result["prepool_groups"])
    selected = max(
        candidates,
        key=lambda item: (
            item["fold0"]["correct"],
            -item["fold0"]["changed_predictions"],
            -item["fold0"]["prepool_groups"],
            item["minimum_similarity"],
        ),
    )
    settings = {
        key: selected[key]
        for key in ("enabled", "maximum_gap", "minimum_similarity", "power")
    }
    selected["fold1"] = run(1, folds[1], records, settings)
    rows = len(folds[0][2]) + len(folds[1][2])
    correct = selected["fold0"]["correct"] + selected["fold1"]["correct"]
    report = {
        "protocol": (
            "Repeat recovery settings are selected on Fold0 from model-probability similarity and acquisition "
            "gaps only. Fold1 is evaluated once blind before any inspection."
        ),
        "selected": selected,
        "combined": {"correct": correct, "rows": rows, "accuracy": correct / rows},
        "deployed_reference": {"correct": 1161, "rows": rows, "accuracy": 1161 / rows},
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "candidates"}, indent=2))


if __name__ == "__main__":
    main()

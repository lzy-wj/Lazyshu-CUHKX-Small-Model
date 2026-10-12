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
from experiments.scene_context_v13.research_pairaware_base_blend import v9_probabilities
from experiments.scene_context_v13.research_sequence_decoder import selected_probabilities
from src.data import FOLDS, get_records


PAIRED_USERS = ({8, 18}, {7, 17})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/scene_context_v13/outputs/domain_base_blend.json",
    )
    return parser.parse_args()


def group_name(user, fold):
    paired = user in PAIRED_USERS[fold]
    environment = int(user > 15)
    return f"{'paired' if paired else 'unmatched'}_env{environment}"


def blend(fold, weights, records):
    old = v9_probabilities(fold)
    new = selected_probabilities(fold)
    if not np.array_equal(old[1], new[1]):
        raise ValueError("V9 and V12 indices do not align")
    values = np.asarray(
        [weights[group_name(records[int(index)]["user"], fold)] for index in old[1]],
        dtype=np.float64,
    )[:, None]
    probabilities = (1 - values) * old[0] + values * new[0]
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities, old[1], old[2]


def score(fold, values, records):
    prepared = prepare_fold(*values, records, PAIR_SOURCES[fold], FOLDS[fold])
    return evaluate(
        prepared,
        template_strength=1.8,
        gap_penalty=0.8,
        maximum_edits=5,
        allowed_users=PAIRED_USERS[fold],
        alignment_beam_size=40,
    )


def main():
    args = parse_args()
    records = get_records()
    groups = ["paired_env0", "paired_env1", "unmatched_env0", "unmatched_env1"]
    candidates = []
    for weight in np.arange(0, 1.001, 0.125):
        weights = {group: float(weight) for group in groups}
        result = score(0, blend(0, weights, records), records)
        candidates.append({"v12_weight": float(weight), "fold0": result})
        print(weight, result["correct"])
    selected_weights = {}
    selected_users = {}
    for user in FOLDS[0]:
        group = group_name(user, 0)
        selected = max(
            candidates,
            key=lambda item: (
                item["fold0"]["per_user"][str(user)]["correct"],
                item["v12_weight"],
            ),
        )
        selected_weights[group] = selected["v12_weight"]
        selected_users[group] = {
            "user": user,
            "correct": selected["fold0"]["per_user"][str(user)]["correct"],
            "rows": selected["fold0"]["per_user"][str(user)]["rows"],
        }
    fold0 = score(0, blend(0, selected_weights, records), records)
    fold1 = score(1, blend(1, selected_weights, records), records)
    rows = len(fold0["per_user"]) and 652 + 705
    correct = fold0["correct"] + fold1["correct"]
    report = {
        "protocol": (
            "A V9/V12 weight is selected independently for each paired-status and environment domain using only "
            "Fold0. The four fixed domain weights are transferred once to Fold1 blind."
        ),
        "selected_weights": selected_weights,
        "fold0_selection_users": selected_users,
        "fold0": fold0,
        "fold1": fold1,
        "combined": {"correct": correct, "rows": rows, "accuracy": correct / rows},
        "v12_reference": {"correct": 1161, "rows": rows, "accuracy": 1161 / rows},
        "global_candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "global_candidates"}, indent=2))


if __name__ == "__main__":
    main()

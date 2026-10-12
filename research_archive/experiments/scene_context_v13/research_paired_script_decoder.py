#!/usr/bin/env python3
import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.research_sequence_decoder import repeat_pool, selected_probabilities
from scripts.evaluate_repeat_decoder import assign_transition_unique, rounds, transition_matrix
from src.data import FOLDS, get_records


PAIR_SOURCES = {
    0: ([18], [8]),
    1: ([17], [7]),
}
TEST_SOURCES = ([20, 21], [16])
TEST_ENVIRONMENT_BOUNDARY = datetime(2025, 6, 9).timestamp()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/scene_context_v13/outputs/paired_script_decoder.json",
    )
    return parser.parse_args()


def user_templates(records, user):
    grouped = defaultdict(list)
    indices = [
        record["index"]
        for record in records
        if record["split"] == "train" and record["user"] == user
    ]
    for item in rounds(indices, records):
        prefix = "-".join(records[item["rows"][0]]["trial"].split("-")[:2])
        grouped[prefix].append(tuple(records[index]["label"] for index in item["rows"]))
    by_length = defaultdict(list)
    for values in grouped.values():
        sequence = Counter(values).most_common(1)[0][0]
        by_length[len(sequence)].append(sequence)
    return by_length


def sequence_score(sequence, unary, transition, transition_strength=0.4):
    positions = np.arange(len(sequence))
    value = float(unary[positions, sequence].sum())
    if len(sequence) > 1:
        value += transition_strength * sum(
            transition[left, right] for left, right in zip(sequence, sequence[1:])
        )
    return value


def evaluate(probabilities, indices, labels, records, sources, excluded_users, bonus_per_token):
    probability_by_index, round_values, groups = repeat_pool(probabilities, indices, records)
    templates = {
        user: user_templates(records, user) for user in {user for values in sources for user in values}
    }
    transitions = [
        transition_matrix(records, set(excluded_users), environment=value) for value in (0, 1)
    ]
    predictions = {
        int(index): int(probability.argmax())
        for index, probability in zip(indices, probabilities)
    }
    template_rounds = 0
    changed = 0
    per_user = defaultdict(lambda: {"rows": 0, "correct": 0})
    for item in round_values:
        user = records[item["rows"][0]]["user"]
        environment = int(user > 15)
        values = np.stack([probability_by_index[index] for index in item["rows"]])
        unary = np.log(np.clip(values, 1e-12, 1.0))
        free = assign_transition_unique(
            values, transitions[environment], 0.4, beam_size=100
        ).argmax(axis=1)
        selected = free
        candidates = [
            sequence
            for source in sources[environment]
            for sequence in templates[source].get(len(item["rows"]), [])
        ]
        if candidates:
            candidate = max(
                candidates,
                key=lambda sequence: sequence_score(
                    sequence, unary, transitions[environment]
                ),
            )
            candidate_score = sequence_score(candidate, unary, transitions[environment])
            free_score = sequence_score(tuple(free), unary, transitions[environment])
            if candidate_score + bonus_per_token * len(candidate) >= free_score:
                selected = np.asarray(candidate)
                template_rounds += 1
        changed += int((selected != free).sum())
        predictions.update(zip(item["rows"], selected.tolist()))
    prediction = np.asarray([predictions[int(index)] for index in indices])
    for index, expected, actual in zip(indices, labels, prediction):
        user = records[int(index)]["user"]
        per_user[user]["rows"] += 1
        per_user[user]["correct"] += int(expected == actual)
    return {
        "correct": int((prediction == labels).sum()),
        "accuracy": float((prediction == labels).mean()),
        "template_rounds": template_rounds,
        "changed_predictions": changed,
        "rounds": len(round_values),
        "repeat_groups": len(groups),
        "per_user": {
            str(user): values | {"accuracy": values["correct"] / values["rows"]}
            for user, values in sorted(per_user.items())
        },
    }


def main():
    args = parse_args()
    records = get_records()
    fold0 = selected_probabilities(0)
    fold1 = selected_probabilities(1)
    candidates = []
    for bonus in np.arange(0, 1.501, 0.025):
        candidates.append(
            {
                "bonus_per_token": float(bonus),
                "fold0": evaluate(
                    *fold0, records, PAIR_SOURCES[0], FOLDS[0], float(bonus)
                ),
            }
        )
    selected = max(
        candidates,
        key=lambda item: (
            item["fold0"]["correct"],
            -item["bonus_per_token"],
        ),
    )
    selected["fold1"] = evaluate(
        *fold1, records, PAIR_SOURCES[1], FOLDS[1], selected["bonus_per_token"]
    )
    combined_correct = selected["fold0"]["correct"] + selected["fold1"]["correct"]
    report = {
        "protocol": (
            "Fold 0 selects only the template acceptance bonus; fold 1 is blind. "
            "Target user identity is hidden from decoding: one paired source template set is "
            "offered to every round in the same environment, including an unmatched control user."
        ),
        "observed_pair_evidence": {
            "pairs": [[6, 16], [7, 17], [8, 18], [9, 19]],
            "test_extrapolation": [[10, 20], [11, 21], [26, 16]],
            "unavailable_pair": [25, 15],
            "test_sources_by_environment": TEST_SOURCES,
        },
        "selected": selected,
        "combined": {
            "correct": combined_correct,
            "rows": len(fold0[2]) + len(fold1[2]),
            "accuracy": combined_correct / (len(fold0[2]) + len(fold1[2])),
        },
        "top_fold0": sorted(
            candidates, key=lambda item: item["fold0"]["correct"], reverse=True
        )[:10],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "top_fold0"}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.research_sequence_decoder import repeat_pool, selected_probabilities
from scripts.evaluate_repeat_decoder import assign_transition_unique, rounds, transition_matrix
from src.data import FOLDS, get_records


def templates(records, excluded_users, environment):
    sequences = Counter()
    users = sorted(
        {
            record["user"]
            for record in records
            if record["split"] == "train"
            and record["user"] not in excluded_users
            and int(record["user"] > 15) == environment
        }
    )
    for user in users:
        grouped = defaultdict(list)
        indices = [
            record["index"]
            for record in records
            if record["split"] == "train" and record["user"] == user
        ]
        for item in rounds(indices, records):
            prefix = "-".join(records[item["rows"][0]]["trial"].split("-")[:2])
            grouped[prefix].append(tuple(records[index]["label"] for index in item["rows"]))
        for values in grouped.values():
            sequences[Counter(values).most_common(1)[0][0]] += 1
    by_length = defaultdict(list)
    for sequence, frequency in sequences.items():
        by_length[len(sequence)].append((sequence, frequency))
    return by_length


def score(sequence, unary, transition, transition_strength):
    positions = np.arange(len(sequence))
    value = float(unary[positions, sequence].sum())
    if len(sequence) > 1:
        value += transition_strength * sum(
            transition[left, right] for left, right in zip(sequence, sequence[1:])
        )
    return value


def evaluate(probabilities, indices, labels, records, fold, bonus_per_token):
    probability_by_index, round_values, groups = repeat_pool(probabilities, indices, records)
    transitions = [
        transition_matrix(records, set(FOLDS[fold]), environment=environment)
        for environment in (0, 1)
    ]
    template_sets = [
        templates(records, set(FOLDS[fold]), environment) for environment in (0, 1)
    ]
    predictions = {int(index): int(probability.argmax()) for index, probability in zip(indices, probabilities)}
    selected_templates = 0
    exact_selected = 0
    exact_available = 0
    selected_correct_delta = 0
    for item in round_values:
        environment = int(records[item["rows"][0]]["user"] > 15)
        values = np.stack([probability_by_index[index] for index in item["rows"]])
        unary = np.log(np.clip(values, 1e-12, 1.0))
        free = assign_transition_unique(
            values, transitions[environment], 0.4, beam_size=100
        ).argmax(axis=1)
        free_score = score(tuple(free), unary, transitions[environment], 0.4)
        candidates = template_sets[environment].get(len(item["rows"]), [])
        true_sequence = tuple(records[index]["label"] for index in item["rows"])
        exact_available += any(sequence == true_sequence for sequence, _ in candidates)
        if candidates:
            template_sequence, frequency = max(
                candidates,
                key=lambda candidate: score(candidate[0], unary, transitions[environment], 0.4)
                + 0.05 * np.log1p(candidate[1]),
            )
            template_score = score(
                template_sequence, unary, transitions[environment], 0.4
            ) + 0.05 * np.log1p(frequency)
            if template_score + bonus_per_token * len(item["rows"]) >= free_score:
                selected_templates += 1
                exact_selected += template_sequence == true_sequence
                selected_correct_delta += sum(
                    int(left == right) - int(base == right)
                    for left, base, right in zip(template_sequence, free, true_sequence)
                )
                free = np.asarray(template_sequence)
        predictions.update(zip(item["rows"], free.tolist()))
    prediction = np.asarray([predictions[int(index)] for index in indices])
    return {
        "correct": int((prediction == labels).sum()),
        "accuracy": float((prediction == labels).mean()),
        "template_rounds": selected_templates,
        "exact_template_selected": exact_selected,
        "exact_template_available": exact_available,
        "selected_correct_delta": selected_correct_delta,
        "rounds": len(round_values),
        "repeat_groups": len(groups),
    }


def main():
    records = get_records()
    fold0 = selected_probabilities(0)
    fold1 = selected_probabilities(1)
    candidates = []
    for bonus in np.arange(0, 2.01, 0.05):
        candidates.append(
            {
                "bonus_per_token": float(bonus),
                "fold0": evaluate(*fold0, records, 0, float(bonus)),
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
        *fold1, records, 1, selected["bonus_per_token"]
    )
    report = {
        "protocol": "Template acceptance bonus selected on fold0; fold1 blind. Templates use only fold-local training users.",
        "selected": selected,
        "combined": {
            "correct": selected["fold0"]["correct"] + selected["fold1"]["correct"],
            "rows": len(fold0[2]) + len(fold1[2]),
        },
        "candidates": candidates,
    }
    report["combined"]["accuracy"] = report["combined"]["correct"] / report["combined"]["rows"]
    output = ROOT / "experiments/scene_context_v13/outputs/template_decoder.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "candidates"}, indent=2))


if __name__ == "__main__":
    main()

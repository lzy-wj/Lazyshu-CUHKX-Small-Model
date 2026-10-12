#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.research_flexible_pair_decoder import (
    evaluate,
    flexible_assignment,
    prepare_fold,
)
from experiments.scene_context_v13.research_pair_identification_decoder import identify_paired_users
from experiments.scene_context_v13.research_paired_script_decoder import PAIR_SOURCES
from experiments.scene_context_v13.research_sequence_decoder import selected_probabilities
from scripts.evaluate_repeat_decoder import rounds
from src.data import FOLDS, get_records


PAIR_CACHE = {}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/scene_context_v13/outputs/roundwise_pair_decoder.json",
    )
    return parser.parse_args()


def source_rounds(records, user):
    indices = [
        record["index"]
        for record in records
        if record["split"] == "train" and record["user"] == user
    ]
    return [tuple(records[index]["label"] for index in item["rows"]) for item in rounds(indices, records)]


def pair_candidates(targets, sources):
    values = [[None for _ in sources] for _ in targets]
    gains = np.full((len(targets), len(sources)), -np.inf, dtype=np.float64)
    for target_index, target in enumerate(targets):
        for source_index, source in enumerate(sources):
            candidate = flexible_assignment(
                target["values"],
                source,
                target["transition"],
                transition_strength=0.4,
                template_strength=1.8,
                gap_penalty=0.8,
                maximum_edits=5,
                beam_size=40,
            )
            if candidate is not None:
                values[target_index][source_index] = candidate
                gains[target_index, source_index] = candidate[0] - target["free_score"]
    return values, gains


def sequence_alignment(candidates, gains, skip_penalty, minimum_gain):
    target_count, source_count = gains.shape
    scores = np.full((target_count + 1, source_count + 1), -np.inf, dtype=np.float64)
    moves = np.zeros((target_count + 1, source_count + 1), dtype=np.int8)
    scores[0, 0] = 0
    scores[1:, 0] = -skip_penalty * np.arange(1, target_count + 1)
    scores[0, 1:] = -skip_penalty * np.arange(1, source_count + 1)
    moves[1:, 0] = 1
    moves[0, 1:] = 2
    for target_index in range(1, target_count + 1):
        for source_index in range(1, source_count + 1):
            match = gains[target_index - 1, source_index - 1] - minimum_gain
            options = (
                scores[target_index - 1, source_index - 1] + match,
                scores[target_index - 1, source_index] - skip_penalty,
                scores[target_index, source_index - 1] - skip_penalty,
            )
            move = int(np.argmax(options))
            scores[target_index, source_index] = options[move]
            moves[target_index, source_index] = move
    matched = []
    target_index, source_index = target_count, source_count
    while target_index or source_index:
        move = int(moves[target_index, source_index])
        if move == 0:
            if candidates[target_index - 1][source_index - 1] is not None:
                matched.append((target_index - 1, source_index - 1))
            target_index -= 1
            source_index -= 1
        elif move == 1:
            target_index -= 1
        else:
            source_index -= 1
    return float(scores[-1, -1]), list(reversed(matched))


def best_cyclic_alignment(targets, sources, skip_penalty, minimum_gain):
    cache_key = (
        tuple(tuple(item["rows"]) for item in targets),
        tuple(sources),
    )
    if cache_key not in PAIR_CACHE:
        PAIR_CACHE[cache_key] = pair_candidates(targets, sources)
    original_candidates, original_gains = PAIR_CACHE[cache_key]
    best = None
    for rotation in range(len(sources)):
        order = list(range(rotation, len(sources))) + list(range(rotation))
        candidates = [[row[index] for index in order] for row in original_candidates]
        gains = original_gains[:, order]
        score, matches = sequence_alignment(candidates, gains, skip_penalty, minimum_gain)
        result = (score, rotation, candidates, matches)
        if best is None or result[0] > best[0]:
            best = result
    return best


def identify_targets(prepared):
    strict = evaluate(prepared, template_strength=0.8, gap_penalty=0.3, maximum_edits=1)
    users, diagnostics = identify_paired_users(prepared, strict)
    mapping = {}
    for user in users:
        environment = next(item["environment"] for item in prepared["rounds"] if item["user"] == user)
        mapping[user] = prepared["sources"][environment][0]
    return mapping, diagnostics


def decode(prepared, records, mapping, skip_penalty, minimum_gain):
    indices = prepared["indices"]
    labels = prepared["labels"]
    probabilities = prepared["probabilities"]
    prediction = {int(index): int(probability.argmax()) for index, probability in zip(indices, probabilities)}
    for item in prepared["rounds"]:
        prediction.update(zip(item["rows"], item["free"]))
    reports = {}
    for user, source in sorted(mapping.items()):
        targets = sorted(
            [item for item in prepared["rounds"] if item["user"] == user],
            key=lambda item: item["start"],
        )
        sources = source_rounds(records, source)
        score, rotation, candidates, matches = best_cyclic_alignment(
            targets, sources, skip_penalty, minimum_gain
        )
        changed = 0
        for target_index, source_index in matches:
            assignment = candidates[target_index][source_index][2]
            target = targets[target_index]
            changed += sum(prediction[index] != label for index, label in zip(target["rows"], assignment))
            prediction.update(zip(target["rows"], assignment))
        reports[str(user)] = {
            "source": source,
            "target_rounds": len(targets),
            "source_rounds": len(sources),
            "rotation": rotation,
            "matches": len(matches),
            "score": score,
            "changed_predictions": changed,
        }
    ordered = np.asarray([prediction[int(index)] for index in indices])
    users = np.asarray([records[int(index)]["user"] for index in indices])
    per_user = {}
    for user in sorted(set(users.tolist())):
        selected = users == user
        per_user[str(user)] = {
            "rows": int(selected.sum()),
            "correct": int((ordered[selected] == labels[selected]).sum()),
            "accuracy": float((ordered[selected] == labels[selected]).mean()),
        }
    return {
        "correct": int((ordered == labels).sum()),
        "accuracy": float((ordered == labels).mean()),
        "changed_predictions": sum(value["changed_predictions"] for value in reports.values()),
        "users": reports,
        "per_user": per_user,
    }


def main():
    args = parse_args()
    records = get_records()
    folds = [selected_probabilities(fold) for fold in (0, 1)]
    prepared = [prepare_fold(*folds[fold], records, PAIR_SOURCES[fold], FOLDS[fold]) for fold in (0, 1)]
    targets = [identify_targets(values) for values in prepared]
    candidates = []
    for skip_penalty in (0.0, 0.5, 1.0, 2.0, 4.0):
        for minimum_gain in (0.0, 2.0, 4.0, 6.0):
            candidates.append(
                {
                    "skip_penalty": skip_penalty,
                    "minimum_gain": minimum_gain,
                    "fold0": decode(prepared[0], records, targets[0][0], skip_penalty, minimum_gain),
                }
            )
    selected = max(
        candidates,
        key=lambda item: (
            item["fold0"]["correct"],
            -item["fold0"]["changed_predictions"],
            -item["skip_penalty"],
            -item["minimum_gain"],
        ),
    )
    selected["fold1"] = decode(
        prepared[1], records, targets[1][0], selected["skip_penalty"], selected["minimum_gain"]
    )
    correct = selected["fold0"]["correct"] + selected["fold1"]["correct"]
    rows = len(folds[0][2]) + len(folds[1][2])
    report = {
        "protocol": (
            "A fixed strict matcher identifies paired targets. Fold0 selects cyclic source-round alignment; "
            "fold1 is evaluated once blind."
        ),
        "identified_targets": [targets[0][0], targets[1][0]],
        "selected": selected,
        "combined": {"correct": correct, "rows": rows, "accuracy": correct / rows},
        "flexible_pair_baseline": {"correct": 1161, "rows": rows, "accuracy": 1161 / rows},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

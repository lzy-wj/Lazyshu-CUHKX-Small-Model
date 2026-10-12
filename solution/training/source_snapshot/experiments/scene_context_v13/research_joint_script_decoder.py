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
from experiments.scene_context_v13.research_ordered_pair_decoder import ordered_templates
from experiments.scene_context_v13.research_pair_identification_decoder import identify_paired_users
from experiments.scene_context_v13.research_paired_script_decoder import PAIR_SOURCES
from experiments.scene_context_v13.research_sequence_decoder import selected_probabilities
from src.data import FOLDS, get_records


PAIR_CACHE = {}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/scene_context_v13/outputs/joint_script_decoder.json",
    )
    return parser.parse_args()


def pair_candidates(targets, templates):
    candidates = [[None for _ in templates] for _ in targets]
    gains = np.full((len(targets), len(templates)), -np.inf, dtype=np.float64)
    for target_index, target in enumerate(targets):
        for template_index, (_, sequence) in enumerate(templates):
            candidate = flexible_assignment(
                target["values"],
                sequence,
                target["transition"],
                transition_strength=0.4,
                template_strength=1.8,
                gap_penalty=0.8,
                maximum_edits=5,
                beam_size=40,
            )
            if candidate is not None:
                candidates[target_index][template_index] = candidate
                gains[target_index, template_index] = candidate[0] - target["free_score"]
    return candidates, gains


def align_rotation(
    candidates,
    gains,
    order,
    minimum_gain,
    repeat_reward,
    target_skip_penalty,
    source_skip_penalty,
):
    target_count = len(candidates)
    source_count = len(order)
    scores = np.full((target_count + 1, source_count + 1), -np.inf, dtype=np.float64)
    moves = [[None for _ in range(source_count + 1)] for _ in range(target_count + 1)]
    scores[0, 0] = 0
    for target_index in range(1, target_count + 1):
        scores[target_index, 0] = scores[target_index - 1, 0] - target_skip_penalty
        moves[target_index][0] = (1, 0, 0)
    for source_index in range(1, source_count + 1):
        scores[0, source_index] = scores[0, source_index - 1] - source_skip_penalty
        moves[0][source_index] = (0, 1, 0)
    for target_index in range(1, target_count + 1):
        for source_index in range(1, source_count + 1):
            options = [
                (scores[target_index - 1, source_index] - target_skip_penalty, 1, 0, 0),
                (scores[target_index, source_index - 1] - source_skip_penalty, 0, 1, 0),
            ]
            template_index = order[source_index - 1]
            for repeats in (1, 2, 3):
                if target_index < repeats:
                    continue
                values = gains[target_index - repeats:target_index, template_index]
                if np.isfinite(values).all():
                    match_score = float(values.sum() - minimum_gain * repeats + repeat_reward * (repeats - 1))
                    options.append(
                        (scores[target_index - repeats, source_index - 1] + match_score, repeats, 1, repeats)
                    )
            value, target_step, source_step, repeats = max(options, key=lambda item: item[0])
            scores[target_index, source_index] = value
            moves[target_index][source_index] = (target_step, source_step, repeats)
    matches = []
    target_index, source_index = target_count, source_count
    while target_index or source_index:
        target_step, source_step, repeats = moves[target_index][source_index]
        if repeats:
            template_index = order[source_index - 1]
            matches.extend(
                (position, template_index)
                for position in range(target_index - repeats, target_index)
            )
        target_index -= target_step
        source_index -= source_step
    return float(scores[-1, -1]), sorted(matches)


def best_alignment(targets, templates, settings):
    key = (tuple(tuple(item["rows"]) for item in targets), tuple(templates))
    if key not in PAIR_CACHE:
        PAIR_CACHE[key] = pair_candidates(targets, templates)
    candidates, gains = PAIR_CACHE[key]
    best = None
    for rotation in range(len(templates)):
        order = list(range(rotation, len(templates))) + list(range(rotation))
        score, matches = align_rotation(candidates, gains, order, **settings)
        result = (score, rotation, matches, candidates)
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


def decode(prepared, records, mapping, settings):
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
        templates = ordered_templates(records, source)
        score, rotation, matches, candidates = best_alignment(targets, templates, settings)
        changed = 0
        for target_index, template_index in matches:
            assignment = candidates[target_index][template_index][2]
            target = targets[target_index]
            changed += sum(prediction[index] != label for index, label in zip(target["rows"], assignment))
            prediction.update(zip(target["rows"], assignment))
        reports[str(user)] = {
            "source": source,
            "target_rounds": len(targets),
            "source_templates": len(templates),
            "rotation": rotation,
            "matched_rounds": len(matches),
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
    for minimum_gain in (0.0, 2.0, 4.0, 6.0):
        for repeat_reward in (0.0, 2.0, 4.0, 6.0):
            for target_skip_penalty in (0.0, 1.0):
                settings = {
                    "minimum_gain": minimum_gain,
                    "repeat_reward": repeat_reward,
                    "target_skip_penalty": target_skip_penalty,
                    "source_skip_penalty": 0.0,
                }
                candidates.append(settings | {"fold0": decode(prepared[0], records, targets[0][0], settings)})
    selected = max(
        candidates,
        key=lambda item: (
            item["fold0"]["correct"],
            -item["fold0"]["changed_predictions"],
            -item["minimum_gain"],
            -item["repeat_reward"],
        ),
    )
    settings = {key: selected[key] for key in (
        "minimum_gain", "repeat_reward", "target_skip_penalty", "source_skip_penalty"
    )}
    selected["fold1"] = decode(prepared[1], records, targets[1][0], settings)
    correct = selected["fold0"]["correct"] + selected["fold1"]["correct"]
    rows = len(folds[0][2]) + len(folds[1][2])
    report = {
        "protocol": (
            "A fixed strict matcher identifies paired targets. Fold0 selects joint cyclic script-order and "
            "one-to-three-repeat alignment; fold1 is evaluated once blind."
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

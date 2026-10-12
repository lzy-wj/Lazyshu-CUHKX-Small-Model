#!/usr/bin/env python3
import argparse
import json
import sys
from collections import Counter, defaultdict
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/scene_context_v13/outputs/ordered_pair_decoder.json",
    )
    return parser.parse_args()


def ordered_templates(records, user):
    indices = [
        record["index"]
        for record in records
        if record["split"] == "train" and record["user"] == user
    ]
    grouped = defaultdict(list)
    first_start = {}
    for item in rounds(indices, records):
        prefix = Counter(
            "-".join(records[index]["trial"].split("-")[:2]) for index in item["rows"]
        ).most_common(1)[0][0]
        grouped[prefix].append(tuple(records[index]["label"] for index in item["rows"]))
        first_start[prefix] = min(first_start.get(prefix, item["start"]), item["start"])
    return [
        (prefix, Counter(grouped[prefix]).most_common(1)[0][0])
        for prefix in sorted(grouped, key=first_start.get)
    ]


def target_units(prepared, user):
    grouped = defaultdict(list)
    for position, item in enumerate(prepared["rounds"]):
        if item["user"] != user:
            continue
        key = ("repeat", item["group"]) if item["group"] is not None else ("single", position)
        grouped[key].append(item)
    return sorted(grouped.values(), key=lambda values: min(item["start"] for item in values))


def unit_candidate(
    unit,
    template,
    template_strength,
    gap_penalty,
    maximum_edits,
    alignment_beam_size,
):
    assignments = []
    gain = 0.0
    for item in unit:
        candidate = flexible_assignment(
            item["values"],
            template,
            item["transition"],
            transition_strength=0.4,
            template_strength=template_strength,
            gap_penalty=gap_penalty,
            maximum_edits=maximum_edits,
            beam_size=alignment_beam_size,
        )
        if candidate is None:
            return None
        gain += float(candidate[0] - item["free_score"])
        assignments.append(candidate[2])
    return gain, assignments


def align_units(units, templates, settings):
    best = None
    for rotation in range(len(templates)):
        ordered = templates[rotation:] + templates[:rotation]
        candidates = [[None for _ in ordered] for _ in units]
        gains = np.full((len(units), len(ordered)), -np.inf, dtype=np.float64)
        for unit_index, unit in enumerate(units):
            for template_index, (_, template) in enumerate(ordered):
                candidate = unit_candidate(
                    unit,
                    template,
                    settings["template_strength"],
                    settings["gap_penalty"],
                    settings["maximum_edits"],
                    settings["alignment_beam_size"],
                )
                if candidate is not None:
                    candidates[unit_index][template_index] = candidate
                    gains[unit_index, template_index] = candidate[0] - settings["minimum_margin"]
        scores = np.zeros((len(units) + 1, len(ordered) + 1), dtype=np.float64)
        moves = np.zeros((len(units) + 1, len(ordered) + 1), dtype=np.int8)
        for unit_index in range(1, len(units) + 1):
            for template_index in range(1, len(ordered) + 1):
                options = (
                    scores[unit_index - 1, template_index],
                    scores[unit_index, template_index - 1],
                    scores[unit_index - 1, template_index - 1] + gains[unit_index - 1, template_index - 1],
                )
                move = int(np.argmax(options))
                scores[unit_index, template_index] = options[move]
                moves[unit_index, template_index] = move
        matched = []
        unit_index, template_index = len(units), len(ordered)
        while unit_index and template_index:
            move = int(moves[unit_index, template_index])
            if move == 0:
                unit_index -= 1
            elif move == 1:
                template_index -= 1
            else:
                candidate = candidates[unit_index - 1][template_index - 1]
                if candidate is not None and gains[unit_index - 1, template_index - 1] > 0:
                    matched.append((unit_index - 1, template_index - 1, candidate))
                unit_index -= 1
                template_index -= 1
        result = (float(scores[-1, -1]), rotation, ordered, list(reversed(matched)))
        if best is None or result[0] > best[0]:
            best = result
    return best


def decode(prepared, records, target_sources, settings):
    probabilities = prepared["probabilities"]
    indices = prepared["indices"]
    labels = prepared["labels"]
    decoded = {int(index): int(probability.argmax()) for index, probability in zip(indices, probabilities)}
    for item in prepared["rounds"]:
        decoded.update(zip(item["rows"], item["free"]))
    user_reports = {}
    for user, source in sorted(target_sources.items()):
        units = target_units(prepared, user)
        templates = ordered_templates(records, source)
        score, rotation, rotated, matched = align_units(units, templates, settings)
        changed = 0
        for unit_index, _, candidate in matched:
            for item, assignment in zip(units[unit_index], candidate[1]):
                changed += sum(decoded[index] != label for index, label in zip(item["rows"], assignment))
                decoded.update(zip(item["rows"], assignment))
        user_reports[str(user)] = {
            "source": source,
            "units": len(units),
            "templates": len(templates),
            "rotation": rotation,
            "matched_units": len(matched),
            "alignment_score": score,
            "changed_predictions": changed,
            "template_order": [prefix for prefix, _ in rotated],
        }
    prediction = np.asarray([decoded[int(index)] for index in indices])
    return {
        "correct": int((prediction == labels).sum()),
        "accuracy": float((prediction == labels).mean()),
        "changed_predictions": sum(values["changed_predictions"] for values in user_reports.values()),
        "users": user_reports,
    }


def identify_targets(prepared):
    strict = evaluate(prepared, template_strength=0.8, gap_penalty=0.3, maximum_edits=1)
    users, diagnostics = identify_paired_users(prepared, strict)
    mapping = {}
    for user in users:
        environment = next(item["environment"] for item in prepared["rounds"] if item["user"] == user)
        mapping[user] = prepared["sources"][environment][0]
    return mapping, diagnostics


def main():
    args = parse_args()
    records = get_records()
    folds = [selected_probabilities(fold) for fold in (0, 1)]
    prepared = [prepare_fold(*folds[fold], records, PAIR_SOURCES[fold], FOLDS[fold]) for fold in (0, 1)]
    targets = [identify_targets(values) for values in prepared]
    candidates = []
    for maximum_edits in (1, 2):
        for gap_penalty in (0.1, 0.3):
            for template_strength in (0.5, 0.65, 0.8):
                for minimum_margin in (0.0, 0.5):
                    settings = {
                        "template_strength": template_strength,
                        "gap_penalty": gap_penalty,
                        "maximum_edits": maximum_edits,
                        "alignment_beam_size": 40,
                        "minimum_margin": minimum_margin,
                    }
                    candidates.append(settings | {"fold0": decode(prepared[0], records, targets[0][0], settings)})
    selected = max(
        candidates,
        key=lambda item: (
            item["fold0"]["correct"],
            -item["fold0"]["changed_predictions"],
            -item["maximum_edits"],
            -item["template_strength"],
        ),
    )
    settings = {key: selected[key] for key in (
        "template_strength", "gap_penalty", "maximum_edits", "alignment_beam_size", "minimum_margin"
    )}
    selected["fold1"] = decode(prepared[1], records, targets[1][0], settings)
    correct = selected["fold0"]["correct"] + selected["fold1"]["correct"]
    rows = len(folds[0][2]) + len(folds[1][2])
    report = {
        "protocol": (
            "A fixed strict matcher identifies paired targets. Fold0 selects cyclic order-preserving script alignment; "
            "fold1 is evaluated once blind."
        ),
        "identified_targets": [targets[0][0], targets[1][0]],
        "identification_diagnostics": [targets[0][1], targets[1][1]],
        "selected": selected,
        "combined": {"correct": correct, "rows": rows, "accuracy": correct / rows},
        "baseline": {"correct": 1103, "rows": rows, "accuracy": 1103 / rows},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

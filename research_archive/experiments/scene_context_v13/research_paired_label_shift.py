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
from experiments.scene_context_v13.research_pairaware_base_blend import PAIRED_USERS
from experiments.scene_context_v13.research_sequence_decoder import selected_probabilities
from src.data import FOLDS, get_records


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/scene_context_v13/outputs/paired_label_shift.json",
    )
    return parser.parse_args()


def source_mapping(fold):
    low_source = PAIR_SOURCES[fold][0][0]
    high_source = PAIR_SOURCES[fold][1][0]
    return {high_source: low_source, low_source: high_source}


def class_counts(records, user):
    counts = np.zeros(40, dtype=np.float64)
    for record in records:
        if record["split"] == "train" and record["user"] == user:
            counts[record["label"]] += 1
    return counts


def adjust(probabilities, indices, records, fold, alpha, power, absent_factor):
    output = probabilities.copy()
    mapping = source_mapping(fold)
    global_counts = sum((class_counts(records, user) for user in mapping), start=np.zeros(40))
    global_prior = (global_counts + alpha) / (global_counts.sum() + 40 * alpha)
    for target, source in mapping.items():
        source_counts = class_counts(records, source)
        source_prior = (source_counts + alpha) / (source_counts.sum() + 40 * alpha)
        multiplier = (source_prior / global_prior) ** power
        multiplier[source_counts == 0] *= absent_factor
        rows = np.asarray([records[int(index)]["user"] == target for index in indices])
        output[rows] *= multiplier
        output[rows] /= output[rows].sum(axis=1, keepdims=True)
    return output


def score(fold, probabilities, indices, labels, records):
    prepared = prepare_fold(
        probabilities,
        indices,
        labels,
        records,
        PAIR_SOURCES[fold],
        FOLDS[fold],
    )
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
    folds = [selected_probabilities(fold) for fold in (0, 1)]
    shortlist = []
    for alpha in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0):
        for power in np.arange(0.0, 1.51, 0.1):
            for absent_factor in (0.1, 0.25, 0.5, 0.75, 1.0):
                adjusted = adjust(*folds[0][:2], records, 0, alpha, float(power), absent_factor)
                prediction = adjusted.argmax(1)
                users = np.asarray([records[int(index)]["user"] for index in folds[0][1]])
                shortlist.append(
                    {
                        "alpha": alpha,
                        "power": float(power),
                        "absent_factor": absent_factor,
                        "raw_correct": int((prediction == folds[0][2]).sum()),
                        "paired_raw_correct": int(
                            ((prediction == folds[0][2]) & np.isin(users, tuple(PAIRED_USERS[0]))).sum()
                        ),
                        "user_raw_correct": {
                            str(user): int(((prediction == folds[0][2]) & (users == user)).sum())
                            for user in PAIRED_USERS[0]
                        },
                    }
                )
    selected_identities = set()
    rankings = (
        sorted(shortlist, key=lambda item: item["raw_correct"], reverse=True)[:10]
        + sorted(shortlist, key=lambda item: item["paired_raw_correct"], reverse=True)[:10]
    )
    for user in PAIRED_USERS[0]:
        rankings += sorted(
            shortlist,
            key=lambda item: item["user_raw_correct"][str(user)],
            reverse=True,
        )[:10]
    for item in rankings:
        selected_identities.add((item["alpha"], item["power"], item["absent_factor"]))
    candidates = []
    for alpha, power, absent_factor in selected_identities:
        adjusted = adjust(*folds[0][:2], records, 0, alpha, power, absent_factor)
        metadata = next(
            item for item in shortlist
            if (item["alpha"], item["power"], item["absent_factor"])
            == (alpha, power, absent_factor)
        )
        candidates.append(
            metadata | {
                "fold0": score(0, adjusted, folds[0][1], folds[0][2], records),
            }
        )
    selected = max(
        candidates,
        key=lambda item: (item["fold0"]["correct"], -item["power"], item["absent_factor"]),
    )
    adjusted = adjust(
        *folds[1][:2],
        records,
        1,
        selected["alpha"],
        selected["power"],
        selected["absent_factor"],
    )
    selected["fold1"] = score(1, adjusted, folds[1][1], folds[1][2], records)
    baseline = [score(fold, *folds[fold], records) for fold in (0, 1)]
    correct = selected["fold0"]["correct"] + selected["fold1"]["correct"]
    report = {
        "protocol": (
            "Fold0 selects a source-label prior correction for paired users only; Fold1 is blind. "
            "Each target uses only the labeled counterpart user's class histogram."
        ),
        "selected": selected,
        "combined": {"correct": correct, "rows": 1357, "accuracy": correct / 1357},
        "baseline": {
            "correct": sum(item["correct"] for item in baseline),
            "rows": 1357,
            "accuracy": sum(item["correct"] for item in baseline) / 1357,
        },
        "top_fold0": sorted(candidates, key=lambda item: item["fold0"]["correct"], reverse=True)[:20],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "top_fold0"}, indent=2))


if __name__ == "__main__":
    main()

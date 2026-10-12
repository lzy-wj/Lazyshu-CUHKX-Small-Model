#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.blend_aware_v12.audit_existing import align, apply_auxiliary, load, temperature_scale
from experiments.scene_context_v13.research_flexible_pair_decoder import evaluate, prepare_fold
from experiments.scene_context_v13.research_paired_script_decoder import PAIR_SOURCES
from experiments.scene_context_v13.research_sequence_decoder import selected_probabilities
from src.data import FOLDS, get_records


PAIRED_USERS = ({8, 18}, {7, 17})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/scene_context_v13/outputs/pairaware_base_blend.json",
    )
    return parser.parse_args()


def v9_probabilities(fold):
    reference = load(ROOT / f"outputs/study_r2plus1d/lightweight_base_int8/base_fold{fold}.npz")
    r2 = align(
        reference,
        load(ROOT / f"outputs/study_r2plus1d/r2_staged_soup_f{fold}_w075_crop075.npz"),
    )
    auxiliary = align(
        reference,
        load(ROOT / f"outputs/study_x3dl_soup/selected_cropblend50_fold{fold}.npz"),
    )
    probabilities = apply_auxiliary(
        0.4 * reference["probabilities"] + 0.6 * temperature_scale(r2, 2.0),
        auxiliary,
    )
    return probabilities, reference["indices"], reference["labels"]


def blended(fold, paired_weight, unmatched_weight, records):
    old = v9_probabilities(fold)
    new = selected_probabilities(fold)
    if not np.array_equal(old[1], new[1]):
        raise ValueError("V9 and V12 indices do not align")
    users = np.asarray([records[int(index)]["user"] for index in old[1]])
    paired = np.isin(users, list(PAIRED_USERS[fold]))
    weights = np.where(paired, paired_weight, unmatched_weight)[:, None]
    probabilities = (1 - weights) * old[0] + weights * new[0]
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities, old[1], old[2]


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
        template_strength=1.6,
        gap_penalty=0.6,
        maximum_edits=5,
        allowed_users=PAIRED_USERS[fold],
        alignment_beam_size=40,
    )


def main():
    args = parse_args()
    records = get_records()
    candidates = []
    for paired_weight in np.arange(0, 1.001, 0.125):
        for unmatched_weight in np.arange(0, 1.001, 0.125):
            probabilities = blended(0, paired_weight, unmatched_weight, records)
            candidates.append(
                {
                    "paired_v12_weight": float(paired_weight),
                    "unmatched_v12_weight": float(unmatched_weight),
                    "fold0": score(0, *probabilities, records),
                }
            )
    selected = max(
        candidates,
        key=lambda item: (
            item["fold0"]["correct"],
            -abs(item["paired_v12_weight"] - item["unmatched_v12_weight"]),
            item["paired_v12_weight"],
            item["unmatched_v12_weight"],
        ),
    )
    fold1 = blended(
        1,
        selected["paired_v12_weight"],
        selected["unmatched_v12_weight"],
        records,
    )
    selected["fold1"] = score(1, *fold1, records)
    correct = selected["fold0"]["correct"] + selected["fold1"]["correct"]
    rows = len(fold1[2]) + 652
    report = {
        "protocol": (
            "Paired and unmatched V9/V12 weights selected jointly on fold0 under the frozen pair-aware decoder; "
            "fold1 is evaluated once blind."
        ),
        "selected": selected,
        "combined": {"correct": correct, "rows": rows, "accuracy": correct / rows},
        "v12_pairaware": {"correct": 1157, "rows": rows, "accuracy": 1157 / rows},
        "top_fold0": sorted(candidates, key=lambda item: item["fold0"]["correct"], reverse=True)[:20],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "top_fold0"}, indent=2))


if __name__ == "__main__":
    main()

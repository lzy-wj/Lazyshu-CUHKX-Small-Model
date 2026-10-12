#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.research_flexible_pair_decoder import flexible_assignment, prepare_fold
from experiments.scene_context_v13.research_paired_script_decoder import PAIR_SOURCES
from experiments.scene_context_v13.research_pairaware_base_blend import PAIRED_USERS
from experiments.scene_context_v13.research_sequence_decoder import selected_probabilities
from src.data import FOLDS, get_records


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    records = get_records()
    probabilities, indices, labels = selected_probabilities(args.fold)
    prepared = prepare_fold(
        probabilities,
        indices,
        labels,
        records,
        PAIR_SOURCES[args.fold],
        FOLDS[args.fold],
    )
    prediction = {int(index): int(values.argmax()) for index, values in zip(indices, probabilities)}
    for item in prepared["rounds"]:
        selected = item["free"]
        best = (item["free_score"], item["free_score"], selected, 0, 0)
        if item["user"] in PAIRED_USERS[args.fold]:
            for source in prepared["sources"][item["environment"]]:
                for template in prepared["templates"][source]:
                    candidate = flexible_assignment(
                        item["values"],
                        template,
                        item["transition"],
                        transition_strength=0.4,
                        template_strength=1.8,
                        gap_penalty=0.8,
                        maximum_edits=5,
                        beam_size=40,
                    )
                    if candidate is not None and candidate[0] > best[0]:
                        best = candidate
            selected = best[2]
        prediction.update(zip(item["rows"], selected))
    pseudo_labels = np.asarray([prediction[int(index)] for index in indices], dtype=np.int64)
    users = np.asarray([records[int(index)]["user"] for index in indices], dtype=np.int64)
    paired = np.isin(users, tuple(PAIRED_USERS[args.fold]))
    output_probabilities = np.full_like(probabilities, 1e-6)
    output_probabilities[np.arange(len(indices)), pseudo_labels] = 1 - 39e-6
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        probabilities=output_probabilities,
        pseudo_labels=pseudo_labels,
        confidence=probabilities.max(1),
        paired=paired,
        indices=indices,
        labels=labels,
    )
    report = {
        "fold": args.fold,
        "rows": len(indices),
        "pseudo_accuracy_audit_only": float((pseudo_labels == labels).mean()),
        "paired_pseudo_accuracy_audit_only": float((pseudo_labels[paired] == labels[paired]).mean()),
        "unmatched_pseudo_accuracy_audit_only": float((pseudo_labels[~paired] == labels[~paired]).mean()),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

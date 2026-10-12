#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.blend_aware_v12.audit_existing import align, load
from experiments.scene_context_v13.evaluate_candidate_domain_pairaware import (
    apply_setting,
    base_probabilities,
    pair_score,
)
from experiments.scene_context_v13.research_domain_base_blend import group_name
from src.data import FOLDS, get_records


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--fold0", type=Path, required=True)
    parser.add_argument("--fold1", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    records = get_records()
    settings_report = json.loads(args.settings.read_text())
    settings = settings_report["domain_settings"]
    outputs = []
    standalone = []
    selected_counts = []
    for fold, path in enumerate((args.fold0, args.fold1)):
        base, indices, labels = base_probabilities(fold, "v13")
        reference = {"probabilities": base, "indices": indices, "labels": labels}
        candidate = align(reference, load(path))
        users = np.asarray([records[int(index)]["user"] for index in indices])
        probabilities = base.copy()
        counts = {}
        for user in FOLDS[fold]:
            group = group_name(user, fold)
            rows = users == user
            probabilities, selected = apply_setting(
                probabilities,
                candidate,
                settings[group],
                rows,
            )
            counts[group] = int(selected.sum())
        outputs.append(pair_score(fold, probabilities, indices, labels, records))
        standalone.append(
            {
                "correct": int((candidate.argmax(axis=1) == labels).sum()),
                "accuracy": float((candidate.argmax(axis=1) == labels).mean()),
            }
        )
        selected_counts.append(counts)
    correct = outputs[0]["correct"] + outputs[1]["correct"]
    rows = sum(len(base_probabilities(fold, "v13")[2]) for fold in (0, 1))
    report = {
        "name": args.name,
        "protocol": "Candidate evaluated once with domain gates frozen from the fixed-soup development report.",
        "settings": str(args.settings),
        "standalone": standalone,
        "selected_rows": selected_counts,
        "fold0": outputs[0],
        "fold1": outputs[1],
        "combined": {"correct": correct, "rows": rows, "accuracy": correct / rows},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

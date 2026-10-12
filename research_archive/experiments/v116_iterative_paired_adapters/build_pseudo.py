#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.research_repeat_recovery_decoder import run
from experiments.v109_crossfit_round_transformer.train import v104_probabilities
from experiments.v114_paired_person_fusion.evaluate import fused_probabilities
from src.data import get_records


PAIR_SETTINGS = {
    "enabled": False,
    "maximum_gap": 120,
    "minimum_similarity": 0.7,
    "power": 1.5,
}
PAIRED_USERS = ({8, 18}, {7, 17})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    records = get_records()
    reference, _, v104, _ = v104_probabilities(args.fold, records)
    probabilities, _ = fused_probabilities(
        args.fold, reference, v104, records, weight=0.6
    )
    decoded = run(
        args.fold,
        (probabilities, reference["indices"], reference["labels"]),
        records,
        PAIR_SETTINGS,
    )
    pseudo_labels = np.asarray(decoded["predictions"], dtype=np.int64)
    users = np.asarray(
        [records[int(index)]["user"] for index in reference["indices"]]
    )
    paired = np.isin(users, tuple(PAIRED_USERS[args.fold]))
    one_hot = np.full_like(probabilities, 1e-6)
    one_hot[np.arange(len(one_hot)), pseudo_labels] = 1 - 39e-6
    output = HERE / "outputs" / f"v116_v114_pseudo_f{args.fold}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        probabilities=one_hot,
        pseudo_labels=pseudo_labels,
        confidence=probabilities.max(axis=1).astype(np.float32),
        paired=paired,
        indices=reference["indices"],
        labels=reference["labels"],
    )
    report = {
        "fold": args.fold,
        "rows": len(pseudo_labels),
        "paired_rows": int(paired.sum()),
        "correct_audit_only": int((pseudo_labels == reference["labels"]).sum()),
        "pseudo_accuracy_audit_only": float(
            (pseudo_labels == reference["labels"]).mean()
        ),
        "teacher": "V114 probabilities plus frozen pair decoder",
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

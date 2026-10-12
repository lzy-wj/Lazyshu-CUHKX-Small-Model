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
from experiments.v84_residual_invariant_skeleton.evaluate import align, load
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
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    records = get_records()
    reference = load(
        ROOT / f"experiments/scene_context_v13/outputs/v23_parameter_soup_warm60_f{args.fold}.npz"
    )
    lowres = align(
        reference,
        load(
            ROOT
            / f"experiments/v24_pseudo_validation/outputs/v23_warm60_pseudo_adapt_f{args.fold}.npz"
        ),
    ).astype(np.float64)
    highres = align(
        reference,
        load(
            ROOT
            / "experiments/v88_highres_pseudo_validation/outputs"
            / f"v23_warm60_pseudo_adapt_highres160_f{args.fold}.npz"
        ),
    ).astype(np.float64)
    probabilities = 0.5 * (lowres + highres)
    decoded = run(
        args.fold,
        (probabilities, reference["indices"], reference["labels"]),
        records,
        PAIR_SETTINGS,
    )
    pseudo_labels = np.asarray(decoded["predictions"], dtype=np.int64)
    users = np.asarray([records[int(index)]["user"] for index in reference["indices"]])
    paired = np.isin(users, tuple(PAIRED_USERS[args.fold]))
    one_hot = np.full_like(probabilities, 1e-6)
    one_hot[np.arange(len(one_hot)), pseudo_labels] = 1 - 39e-6
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
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
        "pseudo_accuracy_audit_only": float((pseudo_labels == reference["labels"]).mean()),
        "correct_audit_only": int((pseudo_labels == reference["labels"]).sum()),
        "teacher": "fixed equal V24-128/V88-160 probabilities plus frozen pair decoder",
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

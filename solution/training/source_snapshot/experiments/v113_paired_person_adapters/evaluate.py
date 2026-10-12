#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.v109_crossfit_round_transformer.train import v104_probabilities
from experiments.v84_residual_invariant_skeleton.evaluate import (
    load,
    pair_metrics,
    raw_metrics,
)
from src.data import get_records


PAIRINGS = ({8: 18, 18: 8}, {7: 17, 17: 7})


def main():
    records = get_records()
    folds = []
    for fold in (0, 1):
        reference, _, v104, _ = v104_probabilities(fold, records)
        positions = {
            int(index): position for position, index in enumerate(reference["indices"])
        }
        personalized = v104.copy()
        adapted_rows = 0
        for target, source in PAIRINGS[fold].items():
            archive = load(
                HERE
                / "outputs"
                / f"v113_paired_s{source}_t{target}_f{fold}"
                / "validation.npz"
            )
            for source_position, index in enumerate(archive["indices"]):
                target_position = positions[int(index)]
                if records[int(index)]["user"] != target:
                    raise ValueError("V113 archive contains the wrong target user")
                personalized[target_position] = 0.5 * (
                    v104[target_position] + archive["probabilities"][source_position]
                )
                adapted_rows += 1
        variants = {"v104": v104, "v113_paired_person": personalized}
        result = {"fold": fold, "adapted_rows": adapted_rows, "variants": {}}
        for name, probabilities in variants.items():
            result["variants"][name] = {
                "raw": raw_metrics(
                    probabilities,
                    reference["indices"],
                    reference["labels"],
                    fold,
                    records,
                ),
                "pair": pair_metrics(
                    probabilities,
                    reference["indices"],
                    reference["labels"],
                    fold,
                    records,
                ),
            }
        folds.append(result)

    summary = {}
    for name in folds[0]["variants"]:
        summary[name] = {}
        for stage in ("raw", "pair"):
            values = [fold["variants"][name][stage] for fold in folds]
            correct = sum(value["correct"] for value in values)
            rows = sum(value["rows"] for value in values)
            weighted = [
                value["test_domain_weighted_accuracy"] for value in values
            ]
            summary[name][stage] = {
                "correct": correct,
                "rows": rows,
                "accuracy": correct / rows,
                "fold_test_domain_weighted_accuracy": weighted,
                "mean_fold_test_domain_weighted_accuracy": float(np.mean(weighted)),
            }
    baseline = summary["v104"]["pair"]
    candidate = summary["v113_paired_person"]["pair"]
    report = {
        "protocol": (
            "Each paired target uses an equal V104 plus target-specific adapter "
            "trained from its true-label counterpart and refreshed target pseudo labels."
        ),
        "pairings": [dict(values) for values in PAIRINGS],
        "folds": folds,
        "summary": summary,
        "promote": candidate["correct"] > baseline["correct"]
        and all(
            new >= old
            for new, old in zip(
                candidate["fold_test_domain_weighted_accuracy"],
                baseline["fold_test_domain_weighted_accuracy"],
            )
        ),
    }
    output = HERE / "outputs" / "evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"summary": summary, "promote": report["promote"]}, indent=2))


if __name__ == "__main__":
    main()

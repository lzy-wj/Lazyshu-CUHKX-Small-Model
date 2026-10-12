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
    align,
    load,
    pair_metrics,
    raw_metrics,
)
from src.data import get_records


def main():
    records = get_records()
    folds = []
    for fold in (0, 1):
        reference, _, v104, domain = v104_probabilities(fold, records)
        balanced = align(
            reference,
            load(
                HERE
                / "outputs"
                / "checkpoints"
                / f"v107_sqrt_class_balance_f{fold}"
                / "validation.npz"
            ),
        )
        target_equal = v104.copy()
        target_equal[domain > 0] = 0.5 * (
            v104[domain > 0] + balanced[domain > 0]
        )
        variants = {
            "v104": v104,
            "v107_balanced": balanced,
            "v107_target_domain_equal": target_equal,
        }
        result = {"fold": fold, "variants": {}}
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
    candidate = summary["v107_target_domain_equal"]["pair"]
    report = {
        "protocol": (
            "Inverse square-root class weighting is the only training change. "
            "The fixed equal ensemble is applied only to domains represented in test."
        ),
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
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"summary": summary, "promote": report["promote"]}, indent=2))


if __name__ == "__main__":
    main()

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


def summarize(folds):
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
    return summary


def main():
    records = get_records()
    folds = []
    for fold in (0, 1):
        reference, _, v104, _ = v104_probabilities(fold, records)
        transformer = align(
            reference,
            load(
                HERE
                / "outputs"
                / f"v109_crossfit_round_transformer_f{fold}"
                / "validation.npz"
            ),
        )
        variants = {
            "v104": v104,
            "v109_crossfit_round_transformer": transformer,
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
    summary = summarize(folds)
    baseline = summary["v104"]["pair"]
    candidate = summary["v109_crossfit_round_transformer"]["pair"]
    report = {
        "protocol": (
            "Each target fold is transformed by a fixed model trained only on the "
            "opposite fold's OOF probabilities and labels; no target-fold selection."
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
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"summary": summary, "promote": report["promote"]}, indent=2))


if __name__ == "__main__":
    main()

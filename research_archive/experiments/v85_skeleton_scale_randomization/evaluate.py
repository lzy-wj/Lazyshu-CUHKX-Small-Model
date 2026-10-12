#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.v84_residual_invariant_skeleton.evaluate import align, load, pair_metrics, raw_metrics
from src.data import get_records


def main():
    records = get_records()
    folds = []
    for fold in (0, 1):
        control = load(
            ROOT
            / f"experiments/scene_context_v13/outputs/checkpoints/"
            f"v23_shared5_joint_warm_maskdistill_f{fold}/validation.npz"
        )
        candidate = load(
            HERE / f"outputs/checkpoints/v85_scale_randomization_isolated_f{fold}/validation.npz"
        )
        candidate_probabilities = align(control, candidate)
        control_probabilities = control["probabilities"].astype(np.float64)
        variants = {
            "control": control_probabilities,
            "scale_randomization": candidate_probabilities,
            "equal_probability_ensemble": 0.5 * control_probabilities + 0.5 * candidate_probabilities,
        }
        fold_results = {"fold": fold, "variants": {}}
        for name, probabilities in variants.items():
            fold_results["variants"][name] = {
                "raw": raw_metrics(probabilities, control["indices"], control["labels"], fold, records),
                "pair": pair_metrics(probabilities, control["indices"], control["labels"], fold, records),
            }
        folds.append(fold_results)
    summary = {}
    for name in folds[0]["variants"]:
        summary[name] = {}
        for stage in ("raw", "pair"):
            values = [fold["variants"][name][stage] for fold in folds]
            summary[name][stage] = {
                "correct": sum(value["correct"] for value in values),
                "rows": sum(value["rows"] for value in values),
                "accuracy": sum(value["correct"] for value in values) / sum(value["rows"] for value in values),
                "mean_fold_test_domain_weighted_accuracy": sum(
                    value["test_domain_weighted_accuracy"] for value in values
                ) / len(values),
                "fold_test_domain_weighted_accuracy": [
                    value["test_domain_weighted_accuracy"] for value in values
                ],
            }
    control = summary["control"]["pair"]
    candidate = summary["scale_randomization"]["pair"]
    report = {
        "protocol": "One-variable training-fold body-scale randomization test on the V23 warm recipe.",
        "folds": folds,
        "summary": summary,
        "promote": (
            candidate["correct"] > control["correct"]
            and all(
                candidate_value > control_value
                for candidate_value, control_value in zip(
                    candidate["fold_test_domain_weighted_accuracy"],
                    control["fold_test_domain_weighted_accuracy"],
                )
            )
        ),
    }
    output = HERE / "outputs/evaluation.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"summary": summary, "promote": report["promote"]}, indent=2))


if __name__ == "__main__":
    main()

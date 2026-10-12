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
        reference = load(
            ROOT / f"experiments/scene_context_v13/outputs/v23_parameter_soup_warm60_f{fold}.npz"
        )
        lowres = align(
            reference,
            load(
                ROOT
                / f"experiments/v24_pseudo_validation/outputs/v23_warm60_pseudo_adapt_f{fold}.npz"
            ),
        )
        highres = align(
            reference,
            load(
                ROOT
                / f"experiments/v88_highres_pseudo_validation/outputs/"
                f"v23_warm60_pseudo_adapt_highres160_f{fold}.npz"
            ),
        )
        consensus = align(reference, load(HERE / f"outputs/v92_consensus_highres_f{fold}.npz"))
        variants = {
            "v24_pseudo_128": lowres,
            "v89_scale_average": 0.5 * (lowres.astype(np.float64) + highres.astype(np.float64)),
            "v92_consensus_160": consensus,
            "v92_consensus_scale_average": 0.5 * (
                lowres.astype(np.float64) + consensus.astype(np.float64)
            ),
        }
        fold_result = {"fold": fold, "variants": {}}
        for name, probabilities in variants.items():
            fold_result["variants"][name] = {
                "raw": raw_metrics(probabilities, reference["indices"], reference["labels"], fold, records),
                "pair": pair_metrics(probabilities, reference["indices"], reference["labels"], fold, records),
            }
        folds.append(fold_result)

    summary = {}
    for name in folds[0]["variants"]:
        summary[name] = {}
        for stage in ("raw", "pair"):
            values = [fold["variants"][name][stage] for fold in folds]
            correct = sum(value["correct"] for value in values)
            rows = sum(value["rows"] for value in values)
            summary[name][stage] = {
                "correct": correct,
                "rows": rows,
                "accuracy": correct / rows,
                "mean_fold_test_domain_weighted_accuracy": sum(
                    value["test_domain_weighted_accuracy"] for value in values
                ) / len(values),
                "fold_test_domain_weighted_accuracy": [
                    value["test_domain_weighted_accuracy"] for value in values
                ],
            }

    control = summary["v89_scale_average"]["pair"]
    candidate = summary["v92_consensus_scale_average"]["pair"]
    report = {
        "protocol": "V88 with original pseudo selection intersected with fixed 128/160 agreement.",
        "folds": folds,
        "summary": summary,
        "promote": candidate["correct"] > control["correct"] and all(
            candidate_value >= control_value
            for candidate_value, control_value in zip(
                candidate["fold_test_domain_weighted_accuracy"],
                control["fold_test_domain_weighted_accuracy"],
            )
        ),
    }
    output = HERE / "outputs/evaluation.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"summary": summary, "promote": report["promote"]}, indent=2))


if __name__ == "__main__":
    main()

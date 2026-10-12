#!/usr/bin/env python3
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.v84_residual_invariant_skeleton.evaluate import align, load, pair_metrics, raw_metrics
from src.data import get_records


def summarize(folds):
    summary = {}
    for name in folds[0]["variants"]:
        summary[name] = {}
        for stage in ("raw", "pair"):
            values = [fold["variants"][name][stage] for fold in folds]
            rows = sum(value["rows"] for value in values)
            correct = sum(value["correct"] for value in values)
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
    return summary


def main():
    records = get_records()
    folds = []
    for fold in (0, 1):
        reference = load(
            ROOT / f"experiments/scene_context_v13/outputs/v23_parameter_soup_warm60_f{fold}.npz"
        )
        highres_name = "v23_highres_base_f0_direct.npz" if fold == 0 else "v23_highres_base_f1.npz"
        variants = {
            "v23_128": reference["probabilities"],
            "v23_160": align(
                reference,
                load(ROOT / "experiments/v86_highres_spatial_encoder/outputs" / highres_name),
            ),
            "v24_pseudo_128": align(
                reference,
                load(
                    ROOT
                    / f"experiments/v24_pseudo_validation/outputs/v23_warm60_pseudo_adapt_f{fold}.npz"
                ),
            ),
            "v88_pseudo_160": align(
                reference,
                load(HERE / f"outputs/v23_warm60_pseudo_adapt_highres160_f{fold}.npz"),
            ),
        }
        fold_result = {"fold": fold, "variants": {}}
        for name, probabilities in variants.items():
            fold_result["variants"][name] = {
                "raw": raw_metrics(probabilities, reference["indices"], reference["labels"], fold, records),
                "pair": pair_metrics(probabilities, reference["indices"], reference["labels"], fold, records),
            }
        folds.append(fold_result)

    summary = summarize(folds)
    control = summary["v24_pseudo_128"]["pair"]
    candidate = summary["v88_pseudo_160"]["pair"]
    promote = candidate["correct"] > control["correct"] and all(
        candidate_value > control_value
        for candidate_value, control_value in zip(
            candidate["fold_test_domain_weighted_accuracy"],
            control["fold_test_domain_weighted_accuracy"],
        )
    )
    report = {
        "protocol": "One-variable 128-to-160 input test on the fixed V24 pseudo-target adaptation recipe.",
        "folds": folds,
        "summary": summary,
        "promotion_rule": (
            "Pair-aware correct must exceed V24 and test-domain-weighted pair accuracy "
            "must improve on both folds."
        ),
        "promote": promote,
    }
    output = HERE / "outputs/evaluation.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"summary": summary, "promote": promote}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import json
import sys
from pathlib import Path


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
        v86_name = (
            "v86_highres160_spatial_masked_final_f0"
            if fold == 0
            else "v86_highres160_spatial_masked_f1"
        )
        variants = {
            "v86_spatial_160": align(
                reference,
                load(
                    ROOT
                    / "experiments/v86_highres_spatial_encoder/outputs"
                    / v86_name
                    / "validation.npz"
                ),
            ),
            "v95_d384": align(
                reference,
                load(HERE / f"outputs/v95_phase_experts_d384_f{fold}/validation.npz"),
            ),
            "v95_d768": align(
                reference,
                load(HERE / f"outputs/v95_phase_experts_d768_f{fold}/validation.npz"),
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

    control = summary["v86_spatial_160"]["pair"]
    promoted = []
    for name in ("v95_d384", "v95_d768"):
        candidate = summary[name]["pair"]
        if candidate["correct"] > control["correct"] and all(
            candidate_value > control_value
            for candidate_value, control_value in zip(
                candidate["fold_test_domain_weighted_accuracy"],
                control["fold_test_domain_weighted_accuracy"],
            )
        ):
            promoted.append(name)
    report = {
        "protocol": "Capacity ablation for fixed phase-aligned semantic label experts at 160px.",
        "folds": folds,
        "summary": summary,
        "promotion_reference": "v86_spatial_160",
        "promoted": promoted,
    }
    output = HERE / "outputs/evaluation.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"summary": summary, "promoted": promoted}, indent=2))


if __name__ == "__main__":
    main()

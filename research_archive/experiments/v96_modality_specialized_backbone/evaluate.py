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


def aggregate(folds):
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
    return summary


def main():
    records = get_records()
    folds = []
    for fold in (0, 1):
        reference = load(
            ROOT / f"experiments/scene_context_v13/outputs/v23_parameter_soup_warm60_f{fold}.npz"
        )
        source_name = "v23_highres_base_f0_direct.npz" if fold == 0 else "v23_highres_base_f1.npz"
        source = align(
            reference,
            load(ROOT / "experiments/v86_highres_spatial_encoder/outputs" / source_name),
        ).astype(np.float64)
        lowres_pseudo = align(
            reference,
            load(
                ROOT
                / f"experiments/v24_pseudo_validation/outputs/v23_warm60_pseudo_adapt_f{fold}.npz"
            ),
        ).astype(np.float64)
        highres_pseudo = align(
            reference,
            load(
                ROOT
                / "experiments/v88_highres_pseudo_validation/outputs"
                / f"v23_warm60_pseudo_adapt_highres160_f{fold}.npz"
            ),
        ).astype(np.float64)
        v89 = 0.5 * (lowres_pseudo + highres_pseudo)
        split4 = align(
            reference,
            load(HERE / f"outputs/v96_modality_specialized_s4_f{fold}/validation.npz"),
        ).astype(np.float64)
        split3 = align(
            reference,
            load(HERE / f"outputs/v96_modality_specialized_s3_f{fold}/validation.npz"),
        ).astype(np.float64)
        specialized = 0.5 * (split4 + split3)
        variants = {
            "v23_highres_source": source,
            "v89_local_best": v89,
            "split4": split4,
            "split3": split3,
            "split4_source_equal": 0.5 * (split4 + source),
            "split3_source_equal": 0.5 * (split3 + source),
            "split4_split3_equal": specialized,
            "split4_v89_equal": 0.5 * (split4 + v89),
            "split3_v89_equal": 0.5 * (split3 + v89),
            "specialized_v89_equal": 0.5 * (specialized + v89),
        }
        fold_result = {"fold": fold, "variants": {}}
        for name, probabilities in variants.items():
            fold_result["variants"][name] = {
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
        folds.append(fold_result)

    summary = aggregate(folds)
    candidates = [name for name in summary if name not in ("v23_highres_source", "v89_local_best")]
    best_name = max(
        candidates,
        key=lambda name: (
            summary[name]["pair"]["correct"],
            summary[name]["pair"]["mean_fold_test_domain_weighted_accuracy"],
            summary[name]["raw"]["correct"],
        ),
    )
    best = summary[best_name]["pair"]
    benchmark = summary["v89_local_best"]["pair"]
    report = {
        "protocol": (
            "Predeclared split-stage ablation and fixed equal-probability ensembles; "
            "no learned or searched ensemble weights."
        ),
        "folds": folds,
        "summary": summary,
        "best_candidate": best_name,
        "beats_v89": (
            best["correct"] > benchmark["correct"]
            and best["mean_fold_test_domain_weighted_accuracy"]
            > benchmark["mean_fold_test_domain_weighted_accuracy"]
        ),
    }
    output = HERE / "outputs/evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "summary": summary,
                "best_candidate": best_name,
                "beats_v89": report["beats_v89"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.v84_residual_invariant_skeleton.evaluate import align, load, pair_metrics, raw_metrics
from experiments.v96_modality_specialized_backbone.evaluate import aggregate
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
        ).astype(np.float64)
        highres = align(
            reference,
            load(
                ROOT
                / "experiments/v88_highres_pseudo_validation/outputs"
                / f"v23_warm60_pseudo_adapt_highres160_f{fold}.npz"
            ),
        ).astype(np.float64)
        v89 = 0.5 * (lowres + highres)
        old_cross = align(
            reference,
            load(
                ROOT
                / "experiments/v98_cross_modal_target_adapter/outputs"
                / f"v98_target_adapter_cross_f{fold}/validation.npz"
            ),
        ).astype(np.float64)
        refreshed_main = align(
            reference,
            load(HERE / f"outputs/v99_refreshed_main_f{fold}/validation.npz"),
        ).astype(np.float64)
        refreshed_cross = align(
            reference,
            load(HERE / f"outputs/v99_refreshed_cross_f{fold}/validation.npz"),
        ).astype(np.float64)
        refreshed_average = 0.5 * (refreshed_main + refreshed_cross)
        variants = {
            "v89_local_best": v89,
            "v98_old_cross": old_cross,
            "refreshed_main": refreshed_main,
            "refreshed_cross": refreshed_cross,
            "refreshed_equal": refreshed_average,
            "refreshed_main_v89_equal": 0.5 * (refreshed_main + v89),
            "refreshed_cross_v89_equal": 0.5 * (refreshed_cross + v89),
            "refreshed_equal_v89_equal": 0.5 * (refreshed_average + v89),
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
    candidates = [name for name in summary if name not in ("v89_local_best", "v98_old_cross")]
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
            "One-variable refresh from V13 to V89 sequence pseudo labels with fixed main/cross "
            "adapter ablation and equal probability ensembles."
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

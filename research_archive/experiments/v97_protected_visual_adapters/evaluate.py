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
        intermediate = align(
            reference,
            load(HERE / f"outputs/v97_protected_visual_intermediate_f{fold}/validation.npz"),
        ).astype(np.float64)
        dual = align(
            reference,
            load(HERE / f"outputs/v97_protected_visual_dual_f{fold}/validation.npz"),
        ).astype(np.float64)
        adapter_average = 0.5 * (intermediate + dual)
        variants = {
            "v23_highres_source": source,
            "v89_local_best": v89,
            "intermediate_adapter": intermediate,
            "dual_adapter": dual,
            "adapter_equal": adapter_average,
            "intermediate_source_equal": 0.5 * (intermediate + source),
            "dual_source_equal": 0.5 * (dual + source),
            "intermediate_v89_equal": 0.5 * (intermediate + v89),
            "dual_v89_equal": 0.5 * (dual + v89),
            "adapter_v89_equal": 0.5 * (adapter_average + v89),
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
            "Predeclared zero-initialized visual-adapter location ablation and fixed equal "
            "probability ensembles; no blend-weight search."
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

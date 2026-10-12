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
        highres_name = "v23_highres_base_f0_direct.npz" if fold == 0 else "v23_highres_base_f1.npz"
        scale128 = reference["probabilities"].astype(np.float64)
        scale160 = align(
            reference,
            load(ROOT / "experiments/v86_highres_spatial_encoder/outputs" / highres_name),
        ).astype(np.float64)
        scale224 = align(
            reference,
            load(HERE / f"outputs/v101_v23_native224_f{fold}.npz"),
        ).astype(np.float64)
        variants = {
            "v23_128": scale128,
            "v23_160": scale160,
            "v23_224": scale224,
            "v23_160_224_equal": 0.5 * (scale160 + scale224),
            "v23_128_160_224_equal": (scale128 + scale160 + scale224) / 3,
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
    report = {
        "protocol": "Frozen V23 native 128/160/224 scale audit with fixed equal scale ensembles.",
        "folds": folds,
        "summary": summary,
        "advance_pseudo_adaptation": all(
            candidate > control
            for candidate, control in zip(
                summary["v23_224"]["pair"]["fold_test_domain_weighted_accuracy"],
                summary["v23_160"]["pair"]["fold_test_domain_weighted_accuracy"],
            )
        ),
    }
    output = HERE / "outputs/direct_evaluation.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"summary": summary, "advance_pseudo_adaptation": report["advance_pseudo_adaptation"]}, indent=2))


if __name__ == "__main__":
    main()

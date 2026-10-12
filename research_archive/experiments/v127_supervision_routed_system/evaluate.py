#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.research_repeat_recovery_decoder import run
from experiments.v109_crossfit_round_transformer.train import v104_probabilities
from experiments.v114_paired_person_fusion.evaluate import fused_probabilities
from experiments.v125_domain_complement.evaluate import scores
from experiments.v70_film_margin_distillation.validate import SETTINGS as V71_SETTINGS
from experiments.v84_residual_invariant_skeleton.evaluate import PAIR_SETTINGS, domains
from src.data import get_records


def main():
    records = get_records()
    output_dir = HERE / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    folds = []
    for fold in (0, 1):
        reference, _, v104, _ = v104_probabilities(fold, records)
        v71_archive = np.load(
            ROOT
            / "experiments/v71_gated_film_distillation/outputs"
            / f"student_f{fold}.npz"
        )
        if not np.array_equal(v71_archive["indices"], reference["indices"]):
            raise ValueError("V71 and V114 validation indices do not align")
        if not np.array_equal(v71_archive["labels"], reference["labels"]):
            raise ValueError("V71 and V114 validation labels do not align")
        v71_result = run(
            fold,
            (
                v71_archive["probabilities"],
                reference["indices"],
                reference["labels"],
            ),
            records,
            V71_SETTINGS,
        )
        v71_prediction = np.asarray(v71_result["predictions"], dtype=np.int64)

        v114_probabilities, _ = fused_probabilities(
            fold, reference, v104, records, weight=0.6
        )
        v114_result = run(
            fold,
            (v114_probabilities, reference["indices"], reference["labels"]),
            records,
            PAIR_SETTINGS,
        )
        v114_prediction = np.asarray(v114_result["predictions"], dtype=np.int64)
        domain = domains(reference["indices"], fold, records)
        prediction = np.where(domain >= 2, v114_prediction, v71_prediction)
        np.savez(
            output_dir / f"v127_validation_f{fold}.npz",
            indices=reference["indices"],
            labels=reference["labels"],
            pseudo_labels=prediction,
        )
        folds.append(
            {
                "fold": fold,
                "v71": scores(v71_prediction, reference, fold, records),
                "v114": scores(v114_prediction, reference, fold, records),
                "v127": scores(prediction, reference, fold, records),
            }
        )

    summary = {}
    for name in ("v71", "v114", "v127"):
        correct = sum(fold[name]["correct"] for fold in folds)
        rows = sum(fold[name]["rows"] for fold in folds)
        weighted = [fold[name]["test_domain_weighted_accuracy"] for fold in folds]
        summary[name] = {
            "correct": correct,
            "rows": rows,
            "accuracy": correct / rows,
            "fold_test_domain_weighted_accuracy": weighted,
            "mean_fold_test_domain_weighted_accuracy": float(np.mean(weighted)),
        }
    report = {
        "protocol": (
            "Frozen causal route: V114 for users with a labeled same-person source; "
            "V71 for unmatched users. No score-dependent gate or fitted weight."
        ),
        "folds": folds,
        "summary": summary,
        "promote": summary["v127"]["correct"] > summary["v71"]["correct"]
        and summary["v127"]["mean_fold_test_domain_weighted_accuracy"]
        > summary["v71"]["mean_fold_test_domain_weighted_accuracy"],
    }
    (output_dir / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"summary": summary, "promote": report["promote"]}, indent=2))


if __name__ == "__main__":
    main()


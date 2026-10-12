#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.research_repeat_recovery_decoder import run
from experiments.scene_context_v13.research_v14_acceptance_gate import (
    accepted,
    fold_values,
)
from experiments.v109_crossfit_round_transformer.train import v104_probabilities
from experiments.v114_paired_person_fusion.evaluate import fused_probabilities
from experiments.v84_residual_invariant_skeleton.evaluate import (
    DOMAIN_WEIGHTS,
    PAIR_SETTINGS,
    domains,
)
from src.data import get_records


V14_PATTERN = (
    "experiments/scene_context_v13/outputs/predictions/"
    "v14_fixedsoup_pseudo_c0p6_e2_composite_f{fold}.npz"
)


def scores(prediction, reference, fold, records):
    domain = domains(reference["indices"], fold, records)
    domain_accuracy = {
        str(value): float(
            (prediction[domain == value] == reference["labels"][domain == value]).mean()
        )
        for value in range(4)
    }
    return {
        "correct": int((prediction == reference["labels"]).sum()),
        "rows": len(prediction),
        "accuracy": float((prediction == reference["labels"]).mean()),
        "domain_accuracy": domain_accuracy,
        "test_domain_weighted_accuracy": sum(
            DOMAIN_WEIGHTS[value] * domain_accuracy[str(value)]
            for value in DOMAIN_WEIGHTS
        ),
    }


def main():
    records = get_records()
    v14_settings = json.loads(
        (
            ROOT
            / "experiments/scene_context_v13/outputs/"
            "shared6_mixup_fixedsoup_uniform_l3int6_l4b0_domain_pairaware.json"
        ).read_text()
    )["domain_settings"]
    selected = json.loads(
        (
            ROOT
            / "experiments/scene_context_v13/outputs/v14_acceptance_gate.json"
        ).read_text()
    )["selected"]
    v14_rule = {
        key: value
        for key, value in selected.items()
        if key not in ("fold0", "fold1", "fold1_audit")
    }
    folds = []
    output_dir = HERE / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    for fold in (0, 1):
        values = fold_values(
            fold,
            ROOT / V14_PATTERN.format(fold=fold),
            v14_settings,
            records,
        )
        use_v14 = (values["old"] != values["new"]) & accepted(values, v14_rule)
        v14_prediction = values["old"].copy()
        v14_prediction[use_v14] = values["new"][use_v14]

        reference, _, v104, _ = v104_probabilities(fold, records)
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
        if not np.array_equal(values["indices"], reference["indices"]):
            raise ValueError("V14 and V114 validation indices do not align")
        if not np.array_equal(values["labels"], reference["labels"]):
            raise ValueError("V14 and V114 validation labels do not align")

        domain = domains(reference["indices"], fold, records)
        prediction = np.where(domain >= 2, v114_prediction, v14_prediction)
        np.savez(
            output_dir / f"v125_validation_f{fold}.npz",
            indices=reference["indices"],
            labels=reference["labels"],
            pseudo_labels=prediction,
        )
        folds.append(
            {
                "fold": fold,
                "v14": scores(v14_prediction, reference, fold, records),
                "v114": scores(v114_prediction, reference, fold, records),
                "v125": scores(prediction, reference, fold, records),
            }
        )

    summary = {}
    for name in ("v14", "v114", "v125"):
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
            "V14 for unmatched users. No score-dependent gate or new fitted weight."
        ),
        "v14_rule": v14_rule,
        "folds": folds,
        "summary": summary,
        "promote": summary["v125"]["correct"] > summary["v114"]["correct"]
        and summary["v125"]["mean_fold_test_domain_weighted_accuracy"]
        > summary["v114"]["mean_fold_test_domain_weighted_accuracy"],
    }
    (output_dir / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"summary": summary, "promote": report["promote"]}, indent=2))


if __name__ == "__main__":
    main()


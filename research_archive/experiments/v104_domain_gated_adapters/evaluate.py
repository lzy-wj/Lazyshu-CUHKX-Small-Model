#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.v84_residual_invariant_skeleton.evaluate import (
    align,
    domains,
    load,
    pair_metrics,
    raw_metrics,
)
from src.data import get_records


def fold_probabilities(fold):
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
    protected = align(
        reference,
        load(
            ROOT
            / "experiments/v97_protected_visual_adapters/outputs"
            / f"v97_protected_visual_dual_f{fold}/validation.npz"
        ),
    )
    old_target = align(
        reference,
        load(
            ROOT
            / "experiments/v98_cross_modal_target_adapter/outputs"
            / f"v98_target_adapter_cross_f{fold}/validation.npz"
        ),
    )
    refreshed_target = align(
        reference,
        load(
            ROOT
            / "experiments/v99_refreshed_sequence_teacher/outputs"
            / f"v99_refreshed_cross_f{fold}/validation.npz"
        ),
    )
    return reference, v89, protected, old_target, refreshed_target


def main():
    records = get_records()
    folds = []
    for fold in (0, 1):
        reference, v89, protected, old_target, refreshed_target = fold_probabilities(fold)
        domain = domains(reference["indices"], fold, records)
        candidate = v89.copy()
        candidate[domain == 1] = (
            v89[domain == 1]
            + protected[domain == 1]
            + old_target[domain == 1]
            + refreshed_target[domain == 1]
        ) / 4
        candidate[domain == 2] = 0.5 * (
            v89[domain == 2] + protected[domain == 2]
        )
        variants = {"v89": v89, "v104_domain_gate": candidate}
        result = {"fold": fold, "variants": {}}
        for name, probabilities in variants.items():
            result["variants"][name] = {
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
        folds.append(result)

    summary = {}
    for name in folds[0]["variants"]:
        values = [fold["variants"][name] for fold in folds]
        summary[name] = {}
        for stage in ("raw", "pair"):
            correct = sum(value[stage]["correct"] for value in values)
            rows = sum(value[stage]["rows"] for value in values)
            summary[name][stage] = {
                "correct": correct,
                "rows": rows,
                "accuracy": correct / rows,
                "fold_test_domain_weighted_accuracy": [
                    value[stage]["test_domain_weighted_accuracy"] for value in values
                ],
                "mean_fold_test_domain_weighted_accuracy": float(
                    np.mean(
                        [value[stage]["test_domain_weighted_accuracy"] for value in values]
                    )
                ),
            }
    baseline = summary["v89"]["pair"]
    candidate = summary["v104_domain_gate"]["pair"]
    report = {
        "protocol": (
            "Fixed equal adapter fusion only in independently supported domains; "
            "all other rows retain V89 exactly."
        ),
        "routing": {
            "domain_1": "equal V89, V97 dual, V98 cross, V99 refreshed cross",
            "domain_2": "equal V89 and V97 dual",
            "domains_0_3": "V89",
        },
        "folds": folds,
        "summary": summary,
        "promote": candidate["correct"] > baseline["correct"]
        and all(
            new > old
            for new, old in zip(
                candidate["fold_test_domain_weighted_accuracy"],
                baseline["fold_test_domain_weighted_accuracy"],
            )
        ),
    }
    output = HERE / "outputs/evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"summary": summary, "promote": report["promote"]}, indent=2))


if __name__ == "__main__":
    main()

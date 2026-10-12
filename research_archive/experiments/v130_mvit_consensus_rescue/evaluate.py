#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.v109_crossfit_round_transformer.train import v104_probabilities
from experiments.v125_domain_complement.evaluate import scores
from experiments.v84_residual_invariant_skeleton.evaluate import align, domains
from src.data import get_records


THRESHOLDS = (0.4, 0.5, 0.6, 0.7)
ROUTES = ("unmatched", "paired", "both")


def load_fold(fold, records):
    reference, _, _, _ = v104_probabilities(fold, records)
    base_archive = np.load(
        ROOT
        / "experiments/v127_supervision_routed_system/outputs"
        / f"v127_validation_f{fold}.npz"
    )
    if not np.array_equal(base_archive["indices"], reference["indices"]):
        raise ValueError("V127 indices do not align")
    conservative = align(
        reference,
        dict(
            np.load(
                ROOT
                / "outputs/checkpoints"
                / f"v126_mvit_full_f{fold}/validation.npz"
            )
        ),
    )
    adapted = align(
        reference,
        dict(
            np.load(
                ROOT
                / "outputs/checkpoints"
                / f"v128_mvit_domainadapt_f{fold}/validation.npz"
            )
        ),
    )
    return reference, base_archive["pseudo_labels"].astype(np.int64), conservative, adapted


def apply_rule(reference, base, conservative, adapted, fold, records, route, threshold):
    first = conservative.argmax(1)
    second = adapted.argmax(1)
    confidence = np.minimum(conservative.max(1), adapted.max(1))
    domain = domains(reference["indices"], fold, records)
    eligible = first == second
    eligible &= confidence >= threshold
    if route == "unmatched":
        eligible &= domain < 2
    elif route == "paired":
        eligible &= domain >= 2
    elif route != "both":
        raise ValueError(route)
    eligible &= first != base
    prediction = base.copy()
    prediction[eligible] = first[eligible]
    labels = reference["labels"]
    return prediction, {
        "changed": int(eligible.sum()),
        "old_correct_on_changes": int((base[eligible] == labels[eligible]).sum()),
        "new_correct_on_changes": int((prediction[eligible] == labels[eligible]).sum()),
    }


def main():
    records = get_records()
    folds = [load_fold(fold, records) for fold in (0, 1)]
    reference0, base0, conservative0, adapted0 = folds[0]
    grid = [
        {
            "route": "control",
            "threshold": 1.0,
            "diagnostics": {"changed": 0},
            "fold0": scores(base0, reference0, 0, records),
        }
    ]
    for route in ROUTES:
        for threshold in THRESHOLDS:
            prediction, diagnostics = apply_rule(
                reference0,
                base0,
                conservative0,
                adapted0,
                0,
                records,
                route,
                threshold,
            )
            grid.append(
                {
                    "route": route,
                    "threshold": threshold,
                    "diagnostics": diagnostics,
                    "fold0": scores(prediction, reference0, 0, records),
                }
            )
    selected = max(
        grid,
        key=lambda item: (
            item["fold0"]["correct"],
            item["fold0"]["test_domain_weighted_accuracy"],
            -item["diagnostics"]["changed"],
        ),
    )
    reference1, base1, conservative1, adapted1 = folds[1]
    if selected["route"] == "control":
        prediction1 = base1
        diagnostics1 = {"changed": 0}
    else:
        prediction1, diagnostics1 = apply_rule(
            reference1,
            base1,
            conservative1,
            adapted1,
            1,
            records,
            selected["route"],
            selected["threshold"],
        )
    selected["fold1_diagnostics"] = diagnostics1
    selected["fold1"] = scores(prediction1, reference1, 1, records)
    base_values = [scores(base0, reference0, 0, records), scores(base1, reference1, 1, records)]
    candidate_values = [selected["fold0"], selected["fold1"]]
    summary = {}
    for name, values in (("v127", base_values), ("v130", candidate_values)):
        correct = sum(value["correct"] for value in values)
        rows = sum(value["rows"] for value in values)
        weighted = [value["test_domain_weighted_accuracy"] for value in values]
        summary[name] = {
            "correct": correct,
            "rows": rows,
            "accuracy": correct / rows,
            "fold_test_domain_weighted_accuracy": weighted,
            "mean_fold_test_domain_weighted_accuracy": float(np.mean(weighted)),
        }
    report = {
        "protocol": (
            "Shared confidence threshold and supervision route selected on Fold0; "
            "transferred unchanged to Fold1. Both MViTs must agree."
        ),
        "thresholds": THRESHOLDS,
        "routes": ROUTES,
        "grid": grid,
        "selected": selected,
        "summary": summary,
        "promote": summary["v130"]["correct"] > summary["v127"]["correct"]
        and summary["v130"]["mean_fold_test_domain_weighted_accuracy"]
        > summary["v127"]["mean_fold_test_domain_weighted_accuracy"],
    }
    output = HERE / "outputs/evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selected": selected, "summary": summary, "promote": report["promote"]}, indent=2))


if __name__ == "__main__":
    main()

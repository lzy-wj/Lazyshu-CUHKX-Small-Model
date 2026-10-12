#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.v125_domain_complement.evaluate import scores
from experiments.v135_end_to_end_label_experts.evaluate import (
    align,
    components,
    decode,
    load,
)
from experiments.v84_residual_invariant_skeleton.evaluate import pair_metrics, raw_metrics
from src.data import get_records


VARIANTS = ("standard", "frozen_visual")
ROUTES = ("unmatched", "paired", "both")
WEIGHTS = (0.02, 0.05, 0.10, 0.20, 0.30, 0.50)
BASELINE_CORRECT = (569, 636)


def candidate_path(variant, fold):
    return (
        HERE
        / "outputs/checkpoints"
        / f"v180_combined_imu_{variant}_f{fold}"
        / "validation.npz"
    )


def evaluate_candidate(fold, variant, route, weight, records):
    reference, v71, v114, domain = components(fold, records)
    candidate = align(reference, load(candidate_path(variant, fold)))
    if route in ("unmatched", "both"):
        v71 = (1 - weight) * v71 + weight * candidate
    if route in ("paired", "both"):
        v114 = (1 - weight) * v114 + weight * candidate
    prediction = decode(fold, reference, v71, v114, domain, records)
    labels = reference["labels"].astype(np.int64)
    incumbent = np.load(
        ROOT
        / "experiments/v127_supervision_routed_system/outputs"
        / f"v127_validation_f{fold}.npz"
    )["pseudo_labels"].astype(np.int64)
    changed = prediction != incumbent
    return {
        "metrics": scores(prediction, reference, fold, records),
        "changes": int(changed.sum()),
        "gains": int((changed & (prediction == labels) & (incumbent != labels)).sum()),
        "losses": int((changed & (prediction != labels) & (incumbent == labels)).sum()),
        "neutral": int((changed & (prediction != labels) & (incumbent != labels)).sum()),
    }


def standalone(fold, variant, records):
    reference, _, _, _ = components(fold, records)
    candidate = align(reference, load(candidate_path(variant, fold)))
    return {
        "raw": raw_metrics(
            candidate, reference["indices"], reference["labels"], fold, records
        ),
        "pair": pair_metrics(
            candidate, reference["indices"], reference["labels"], fold, records
        ),
    }


def main():
    records = get_records()
    variants = {}
    for variant in VARIANTS:
        fold0_grid = []
        for route in ROUTES:
            for weight in WEIGHTS:
                fold0_grid.append(
                    {
                        "route": route,
                        "weight": weight,
                        "result": evaluate_candidate(
                            0, variant, route, weight, records
                        ),
                    }
                )
        selected = max(
            fold0_grid,
            key=lambda item: (
                item["result"]["metrics"]["correct"],
                -item["result"]["losses"],
                -item["result"]["changes"],
                -item["weight"],
            ),
        )
        fold1 = evaluate_candidate(
            1,
            variant,
            selected["route"],
            selected["weight"],
            records,
        )
        fold0_correct = selected["result"]["metrics"]["correct"]
        fold1_correct = fold1["metrics"]["correct"]
        variants[variant] = {
            "standalone": {
                "fold0": standalone(0, variant, records),
                "fold1": standalone(1, variant, records),
            },
            "fold0_grid": fold0_grid,
            "selected": selected | {"fold1": fold1},
            "summary": {
                "correct": fold0_correct + fold1_correct,
                "rows": 1357,
                "accuracy": (fold0_correct + fold1_correct) / 1357,
            },
            "promote": (
                fold0_correct >= BASELINE_CORRECT[0]
                and fold1_correct >= BASELINE_CORRECT[1]
                and fold0_correct + fold1_correct > sum(BASELINE_CORRECT)
            ),
        }
    report = {
        "protocol": (
            "Variant, route, and blend weight are selected on Fold0. The exact route "
            "and weight are transferred once to Fold1 under the frozen V127 decoder."
        ),
        "v127": {"fold0": 569, "fold1": 636, "correct": 1205, "rows": 1357},
        "variants": variants,
    }
    output = HERE / "outputs/evaluation.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        name: {
            "selected": values["selected"],
            "summary": values["summary"],
            "promote": values["promote"],
        }
        for name, values in variants.items()
    }, indent=2))


if __name__ == "__main__":
    main()

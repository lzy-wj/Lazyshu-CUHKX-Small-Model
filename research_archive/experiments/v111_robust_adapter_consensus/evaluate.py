#!/usr/bin/env python3
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.v104_domain_gated_adapters.evaluate import fold_probabilities
from experiments.v84_residual_invariant_skeleton.evaluate import (
    domains,
    pair_metrics,
    raw_metrics,
)
from src.data import get_records


def normalize(values):
    values = np.clip(values, 1e-12, None)
    return values / values.sum(axis=1, keepdims=True)


def pool(values, mode, anchor):
    stacked = np.stack(values, axis=0).astype(np.float64)
    if mode == "arithmetic":
        return stacked.mean(axis=0)
    if mode == "geometric":
        return normalize(np.exp(np.log(np.clip(stacked, 1e-12, 1)).mean(axis=0)))
    if mode == "median":
        return normalize(np.median(stacked, axis=0))
    if mode == "strict_majority":
        output = anchor.copy()
        predictions = stacked.argmax(axis=2)
        required = len(values) // 2 + 1
        averaged = stacked.mean(axis=0)
        for row in range(stacked.shape[1]):
            label, count = Counter(predictions[:, row].tolist()).most_common(1)[0]
            if count >= required:
                output[row] = averaged[row]
        return output
    raise ValueError(f"Unknown aggregation mode: {mode}")


def candidate_probabilities(fold, mode, records):
    reference, v89, protected, old_target, refreshed_target = fold_probabilities(fold)
    domain = domains(reference["indices"], fold, records)
    candidate = v89.copy()
    selected = domain == 1
    candidate[selected] = pool(
        [
            v89[selected],
            protected[selected],
            old_target[selected],
            refreshed_target[selected],
        ],
        mode,
        v89[selected],
    )
    selected = domain == 2
    candidate[selected] = pool(
        [v89[selected], protected[selected]], mode, v89[selected]
    )
    return reference, v89, candidate


def metrics(probabilities, reference, fold, records):
    return {
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


def main():
    records = get_records()
    modes = ("arithmetic", "geometric", "median", "strict_majority")
    fold0 = {"fold": 0, "variants": {}}
    for mode in modes:
        reference, _, candidate = candidate_probabilities(0, mode, records)
        fold0["variants"][mode] = metrics(candidate, reference, 0, records)
    selected_mode = max(
        modes,
        key=lambda mode: (
            fold0["variants"][mode]["pair"]["test_domain_weighted_accuracy"],
            fold0["variants"][mode]["pair"]["correct"],
            -modes.index(mode),
        ),
    )
    reference, _, fold1_candidate = candidate_probabilities(1, selected_mode, records)
    _, _, fold1_arithmetic = candidate_probabilities(1, "arithmetic", records)
    fold1 = {
        "fold": 1,
        "variants": {
            "arithmetic": metrics(fold1_arithmetic, reference, 1, records),
            selected_mode: metrics(fold1_candidate, reference, 1, records),
        },
    }
    incumbent_correct = (
        fold0["variants"]["arithmetic"]["pair"]["correct"]
        + fold1["variants"]["arithmetic"]["pair"]["correct"]
    )
    candidate_correct = (
        fold0["variants"][selected_mode]["pair"]["correct"]
        + fold1["variants"][selected_mode]["pair"]["correct"]
    )
    incumbent_weighted = [
        fold0["variants"]["arithmetic"]["pair"]["test_domain_weighted_accuracy"],
        fold1["variants"]["arithmetic"]["pair"]["test_domain_weighted_accuracy"],
    ]
    candidate_weighted = [
        fold0["variants"][selected_mode]["pair"]["test_domain_weighted_accuracy"],
        fold1["variants"][selected_mode]["pair"]["test_domain_weighted_accuracy"],
    ]
    report = {
        "protocol": (
            "Select one standard robust aggregation rule on Fold 0 target-domain-weighted "
            "pair accuracy; evaluate that frozen rule once on Fold 1."
        ),
        "selected_mode": selected_mode,
        "fold0": fold0,
        "fold1": fold1,
        "summary": {
            "v104_arithmetic": {
                "pair_correct": incumbent_correct,
                "rows": 1357,
                "mean_fold_test_domain_weighted_accuracy": float(
                    np.mean(incumbent_weighted)
                ),
            },
            "v111_selected": {
                "mode": selected_mode,
                "pair_correct": candidate_correct,
                "rows": 1357,
                "mean_fold_test_domain_weighted_accuracy": float(
                    np.mean(candidate_weighted)
                ),
            },
        },
        "promote": candidate_correct > incumbent_correct
        and all(new >= old for new, old in zip(candidate_weighted, incumbent_weighted)),
    }
    output = HERE / "outputs" / "evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "selected_mode": selected_mode,
                "summary": report["summary"],
                "promote": report["promote"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

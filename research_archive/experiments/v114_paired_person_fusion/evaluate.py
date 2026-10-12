#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.v109_crossfit_round_transformer.train import v104_probabilities
from experiments.v84_residual_invariant_skeleton.evaluate import (
    load,
    pair_metrics,
    raw_metrics,
)
from src.data import get_records


PAIRINGS = ({8: 18, 18: 8}, {7: 17, 17: 7})
WEIGHTS = (0.2, 0.4, 0.6, 0.8)


def fused_probabilities(fold, reference, baseline, records, weight):
    positions = {
        int(index): position for position, index in enumerate(reference["indices"])
    }
    fused = baseline.copy()
    adapted_rows = 0
    for target, source in PAIRINGS[fold].items():
        archive = load(
            ROOT
            / "experiments/v113_paired_person_adapters/outputs"
            / f"v113_paired_s{source}_t{target}_f{fold}"
            / "validation.npz"
        )
        for source_position, index in enumerate(archive["indices"]):
            target_position = positions[int(index)]
            if records[int(index)]["user"] != target:
                raise ValueError("V113 archive contains the wrong target user")
            fused[target_position] = (
                (1.0 - weight) * baseline[target_position]
                + weight * archive["probabilities"][source_position]
            )
            adapted_rows += 1
    return fused, adapted_rows


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
    references = []
    baselines = []
    for fold in (0, 1):
        reference, _, baseline, _ = v104_probabilities(fold, records)
        references.append(reference)
        baselines.append(baseline)

    fold0_grid = []
    for weight in WEIGHTS:
        probabilities, adapted_rows = fused_probabilities(
            0, references[0], baselines[0], records, weight
        )
        values = metrics(probabilities, references[0], 0, records)
        fold0_grid.append(
            {"adapter_weight": weight, "adapted_rows": adapted_rows, **values}
        )
    selected = max(
        fold0_grid,
        key=lambda item: (
            item["pair"]["correct"],
            item["pair"]["test_domain_weighted_accuracy"],
            -abs(item["adapter_weight"] - 0.5),
        ),
    )
    weight = selected["adapter_weight"]
    candidate_folds = [selected]
    fold1_probabilities, adapted_rows = fused_probabilities(
        1, references[1], baselines[1], records, weight
    )
    candidate_folds.append(
        {
            "adapter_weight": weight,
            "adapted_rows": adapted_rows,
            **metrics(fold1_probabilities, references[1], 1, records),
        }
    )
    baseline_folds = [
        metrics(baselines[fold], references[fold], fold, records)
        for fold in (0, 1)
    ]

    summary = {}
    for name, folds in (("v104", baseline_folds), ("v114", candidate_folds)):
        summary[name] = {}
        for stage in ("raw", "pair"):
            correct = sum(values[stage]["correct"] for values in folds)
            rows = sum(values[stage]["rows"] for values in folds)
            weighted = [
                values[stage]["test_domain_weighted_accuracy"] for values in folds
            ]
            summary[name][stage] = {
                "correct": correct,
                "rows": rows,
                "accuracy": correct / rows,
                "fold_test_domain_weighted_accuracy": weighted,
                "mean_fold_test_domain_weighted_accuracy": float(np.mean(weighted)),
            }
    baseline = summary["v104"]["pair"]
    candidate = summary["v114"]["pair"]
    report = {
        "protocol": (
            "Adapter weight is selected from a coarse fixed grid on Fold0; "
            "the selected value is applied unchanged to Fold1."
        ),
        "weights": WEIGHTS,
        "fold0_grid": fold0_grid,
        "selected_adapter_weight": weight,
        "candidate_folds": candidate_folds,
        "baseline_folds": baseline_folds,
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
    print(
        json.dumps(
            {
                "selected_adapter_weight": weight,
                "summary": summary,
                "promote": report["promote"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

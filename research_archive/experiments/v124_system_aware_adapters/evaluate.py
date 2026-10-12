#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.v109_crossfit_round_transformer.train import v104_probabilities
from experiments.v114_paired_person_fusion.evaluate import fused_probabilities
from experiments.v84_residual_invariant_skeleton.evaluate import (
    load,
    pair_metrics,
    raw_metrics,
)
from src.data import get_records


PAIRINGS = ({8: 18, 18: 8}, {7: 17, 17: 7})
WEIGHTS = (0.2, 0.4, 0.6, 0.8)


def system_fusion(fold, reference, baseline, records, weight):
    positions = {
        int(index): position for position, index in enumerate(reference["indices"])
    }
    probabilities = baseline.copy()
    for target, source in PAIRINGS[fold].items():
        archive = load(
            HERE
            / "outputs"
            / f"v124_system_s{source}_t{target}_f{fold}"
            / "validation.npz"
        )
        for source_position, index in enumerate(archive["indices"]):
            position = positions[int(index)]
            if records[int(index)]["user"] != target:
                raise ValueError("Target user mismatch")
            probabilities[position] = (
                (1 - weight) * baseline[position]
                + weight * archive["probabilities"][source_position]
            )
    return probabilities


def score(probabilities, reference, fold, records):
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


def summarize(folds):
    output = {}
    for stage in ("raw", "pair"):
        correct = sum(item[stage]["correct"] for item in folds)
        rows = sum(item[stage]["rows"] for item in folds)
        weighted = [item[stage]["test_domain_weighted_accuracy"] for item in folds]
        output[stage] = {
            "correct": correct,
            "rows": rows,
            "accuracy": correct / rows,
            "fold_test_domain_weighted_accuracy": weighted,
            "mean_fold_test_domain_weighted_accuracy": float(np.mean(weighted)),
        }
    return output


def main():
    records = get_records()
    references = []
    v104_values = []
    controls = []
    for fold in (0, 1):
        reference, _, v104, _ = v104_probabilities(fold, records)
        control, _ = fused_probabilities(fold, reference, v104, records, 0.6)
        references.append(reference)
        v104_values.append(v104)
        controls.append(control)
    grid = []
    for weight in WEIGHTS:
        probabilities = system_fusion(
            0, references[0], v104_values[0], records, weight
        )
        grid.append(
            {"adapter_weight": weight, **score(probabilities, references[0], 0, records)}
        )
    selected = max(
        grid,
        key=lambda item: (
            item["pair"]["correct"],
            item["pair"]["test_domain_weighted_accuracy"],
            -item["adapter_weight"],
        ),
    )
    weight = selected["adapter_weight"]
    candidates = [
        selected,
        score(
            system_fusion(1, references[1], v104_values[1], records, weight),
            references[1],
            1,
            records,
        ),
    ]
    control_scores = [
        score(controls[fold], references[fold], fold, records) for fold in (0, 1)
    ]
    summary = {"v114": summarize(control_scores), "v124": summarize(candidates)}
    report = {
        "protocol": (
            "V104 soft-target KL is fixed before training. Fold0 selects a "
            "coarse adapter weight and Fold1 uses it unchanged."
        ),
        "fold0_grid": grid,
        "selected_adapter_weight": weight,
        "candidate_folds": candidates,
        "control_folds": control_scores,
        "summary": summary,
        "promote": summary["v124"]["pair"]["correct"]
        > summary["v114"]["pair"]["correct"]
        and all(
            new > old
            for new, old in zip(
                summary["v124"]["pair"]["fold_test_domain_weighted_accuracy"],
                summary["v114"]["pair"]["fold_test_domain_weighted_accuracy"],
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

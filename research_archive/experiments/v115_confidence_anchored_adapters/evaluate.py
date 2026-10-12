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


def fused(fold, reference, baseline, records, weight):
    positions = {
        int(index): position for position, index in enumerate(reference["indices"])
    }
    probabilities = baseline.copy()
    for target, source in PAIRINGS[fold].items():
        archive = load(
            HERE
            / "outputs"
            / f"v115_anchored_s{source}_t{target}_f{fold}"
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
    baselines = []
    for fold in (0, 1):
        reference, _, baseline, _ = v104_probabilities(fold, records)
        references.append(reference)
        baselines.append(baseline)
    grid = []
    for weight in WEIGHTS:
        probabilities = fused(0, references[0], baselines[0], records, weight)
        grid.append({"adapter_weight": weight, **score(probabilities, references[0], 0, records)})
    selected = max(
        grid,
        key=lambda item: (
            item["pair"]["correct"],
            item["pair"]["test_domain_weighted_accuracy"],
            -abs(item["adapter_weight"] - 0.5),
        ),
    )
    weight = selected["adapter_weight"]
    candidates = [
        selected,
        score(fused(1, references[1], baselines[1], records, weight), references[1], 1, records),
    ]
    controls = [
        score(baselines[fold], references[fold], fold, records) for fold in (0, 1)
    ]
    summary = {"v104": summarize(controls), "v115": summarize(candidates)}
    report = {
        "protocol": (
            "Confidence-anchored training is fixed before evaluation. Fold0 "
            "selects a coarse fusion weight and Fold1 confirms it unchanged."
        ),
        "fold0_grid": grid,
        "selected_adapter_weight": weight,
        "candidate_folds": candidates,
        "control_folds": controls,
        "summary": summary,
        "promote": summary["v115"]["pair"]["correct"]
        > summary["v104"]["pair"]["correct"]
        and all(
            new > old
            for new, old in zip(
                summary["v115"]["pair"]["fold_test_domain_weighted_accuracy"],
                summary["v104"]["pair"]["fold_test_domain_weighted_accuracy"],
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

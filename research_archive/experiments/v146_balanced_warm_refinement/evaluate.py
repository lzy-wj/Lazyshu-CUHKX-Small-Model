#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.v125_domain_complement.evaluate import scores
from experiments.v135_end_to_end_label_experts.evaluate import align, components, decode, load
from experiments.v84_residual_invariant_skeleton.evaluate import pair_metrics, raw_metrics
from src.data import get_records


ROUTES = ("unmatched", "paired", "both")
WEIGHTS = (0.05, 0.10, 0.20, 0.30)


def candidate(fold, values, route, weight, records):
    reference, v71, v114, domain, balanced = values
    if route in ("unmatched", "both"):
        v71 = (1 - weight) * v71 + weight * balanced
    if route in ("paired", "both"):
        v114 = (1 - weight) * v114 + weight * balanced
    prediction = decode(fold, reference, v71, v114, domain, records)
    return scores(prediction, reference, fold, records)


def main():
    records = get_records()
    folds = {}
    standalone = {}
    for fold in (0, 1):
        reference, v71, v114, domain = components(fold, records)
        balanced = align(
            reference,
            load(
                HERE
                / f"outputs/checkpoints/v146_balanced_f{fold}/validation.npz"
            ),
        )
        folds[fold] = (reference, v71, v114, domain, balanced)
        standalone[str(fold)] = {
            "raw": raw_metrics(
                balanced, reference["indices"], reference["labels"], fold, records
            ),
            "pair": pair_metrics(
                balanced, reference["indices"], reference["labels"], fold, records
            ),
        }
    reference0, v710, v1140, domain0, _ = folds[0]
    control0 = scores(
        decode(0, reference0, v710, v1140, domain0, records),
        reference0,
        0,
        records,
    )
    grid = [{"route": "control", "weight": 0.0, "fold0": control0}]
    for route in ROUTES:
        for weight in WEIGHTS:
            grid.append(
                {
                    "route": route,
                    "weight": weight,
                    "fold0": candidate(0, folds[0], route, weight, records),
                }
            )
    selected = max(
        grid,
        key=lambda item: (
            item["fold0"]["correct"],
            item["fold0"]["test_domain_weighted_accuracy"],
            -item["weight"],
        ),
    )
    if selected["route"] == "control":
        reference1, v711, v1141, domain1, _ = folds[1]
        prediction1 = decode(1, reference1, v711, v1141, domain1, records)
        fold1 = scores(prediction1, reference1, 1, records)
    else:
        fold1 = candidate(1, folds[1], selected["route"], selected["weight"], records)
    correct = selected["fold0"]["correct"] + fold1["correct"]
    rows = selected["fold0"]["rows"] + fold1["rows"]
    report = {
        "protocol": "Route and coarse blend weight selected on Fold0, then transferred to Fold1.",
        "standalone": standalone,
        "grid": grid,
        "selected": selected | {"fold1": fold1},
        "summary": {"correct": correct, "rows": rows, "accuracy": correct / rows},
        "v127": {"correct": 1205, "rows": 1357},
        "promote": correct > 1205,
    }
    output = HERE / "outputs/evaluation.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"standalone": standalone, "selected": report["selected"], "summary": report["summary"], "promote": report["promote"]}, indent=2))


if __name__ == "__main__":
    main()


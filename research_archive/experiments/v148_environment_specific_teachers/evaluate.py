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


WEIGHTS = (0.05, 0.10, 0.20, 0.40, 0.60)


def routed_specialist(fold, reference, records):
    users = np.asarray([records[int(index)]["user"] for index in reference["indices"]])
    output = np.zeros((len(users), 40), dtype=np.float64)
    diagnostics = {}
    for environment in (0, 1):
        archive = load(
            HERE
            / f"outputs/checkpoints/v148_env{environment}_f{fold}/validation.npz"
        )
        probabilities = align(reference, archive)
        selected = (users > 15) == bool(environment)
        output[selected] = probabilities[selected]
        diagnostics[str(environment)] = {
            "raw_correct": int(
                (
                    probabilities[selected].argmax(axis=1)
                    == reference["labels"][selected]
                ).sum()
            ),
            "rows": int(selected.sum()),
        }
    return output, diagnostics


def candidate(fold, values, weight, records):
    reference, v71, v114, domain, specialist = values
    v71 = (1 - weight) * v71 + weight * specialist
    v114 = (1 - weight) * v114 + weight * specialist
    prediction = decode(fold, reference, v71, v114, domain, records)
    return scores(prediction, reference, fold, records)


def main():
    records = get_records()
    folds = {}
    diagnostics = {}
    for fold in (0, 1):
        reference, v71, v114, domain = components(fold, records)
        specialist, fold_diagnostics = routed_specialist(fold, reference, records)
        folds[fold] = (reference, v71, v114, domain, specialist)
        diagnostics[str(fold)] = fold_diagnostics | {
            "raw": raw_metrics(
                specialist, reference["indices"], reference["labels"], fold, records
            ),
            "pair": pair_metrics(
                specialist, reference["indices"], reference["labels"], fold, records
            ),
        }
    reference0, v710, v1140, domain0, _ = folds[0]
    control0 = scores(
        decode(0, reference0, v710, v1140, domain0, records),
        reference0,
        0,
        records,
    )
    grid = [{"weight": 0.0, "fold0": control0}]
    for weight in WEIGHTS:
        grid.append({"weight": weight, "fold0": candidate(0, folds[0], weight, records)})
    selected = max(
        grid,
        key=lambda item: (
            item["fold0"]["correct"],
            item["fold0"]["test_domain_weighted_accuracy"],
            -item["weight"],
        ),
    )
    fold1 = candidate(1, folds[1], selected["weight"], records)
    correct = selected["fold0"]["correct"] + fold1["correct"]
    rows = selected["fold0"]["rows"] + fold1["rows"]
    report = {
        "protocol": "One shared specialist weight selected on Fold0 and transferred to Fold1.",
        "diagnostics": diagnostics,
        "grid": grid,
        "selected": selected | {"fold1": fold1},
        "summary": {"correct": correct, "rows": rows, "accuracy": correct / rows},
        "v127": {"correct": 1205, "rows": 1357},
        "promote": correct > 1205,
    }
    output = HERE / "outputs/evaluation.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"diagnostics": diagnostics, "selected": report["selected"], "summary": report["summary"], "promote": report["promote"]}, indent=2))


if __name__ == "__main__":
    main()


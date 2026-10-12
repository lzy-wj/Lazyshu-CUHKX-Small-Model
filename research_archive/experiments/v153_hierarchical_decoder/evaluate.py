#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.v125_domain_complement.evaluate import scores
from experiments.v135_end_to_end_label_experts.evaluate import components, decode
from src.data import get_records


CATEGORIES = ((0, 6), (6, 12), (12, 17), (17, 23), (23, 28), (28, 37), (37, 40))


def reinforce(probabilities):
    category_mass = np.stack(
        [probabilities[:, start:stop].sum(axis=1) for start, stop in CATEGORIES],
        axis=1,
    )
    label_category = np.concatenate(
        [np.full(stop - start, category) for category, (start, stop) in enumerate(CATEGORIES)]
    )
    output = probabilities * category_mass[:, label_category]
    return output / output.sum(axis=1, keepdims=True).clip(1e-12)


def evaluate_fold(fold, records):
    reference, v71, v114, domain = components(fold, records)
    prediction = decode(
        fold,
        reference,
        reinforce(v71),
        reinforce(v114),
        domain,
        records,
    )
    return scores(prediction, reference, fold, records)


def main():
    records = get_records()
    fold0 = evaluate_fold(0, records)
    control0 = json.loads(
        (ROOT / "experiments/v127_supervision_routed_system/outputs/evaluation.json").read_text()
    )["folds"][0]["v127"]
    advance = (
        fold0["correct"] > control0["correct"]
        and fold0["test_domain_weighted_accuracy"]
        > control0["test_domain_weighted_accuracy"]
    )
    report = {
        "protocol": "Fixed category evidence product; Fold1 opens only after a dual Fold0 gain.",
        "categories": [list(values) for values in CATEGORIES],
        "fold0": fold0,
        "control_fold0": control0,
        "advance": advance,
    }
    if advance:
        fold1 = evaluate_fold(1, records)
        report["fold1"] = fold1
        report["summary"] = {
            "correct": fold0["correct"] + fold1["correct"],
            "rows": fold0["rows"] + fold1["rows"],
        }
        report["promote"] = report["summary"]["correct"] > 1205
    output = HERE / "outputs/evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

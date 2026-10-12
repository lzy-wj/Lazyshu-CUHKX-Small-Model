#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.research_repeat_recovery_decoder import run
from experiments.v109_crossfit_round_transformer.train import v104_probabilities
from experiments.v114_paired_person_fusion.evaluate import fused_probabilities
from experiments.v125_domain_complement.evaluate import scores
from experiments.v70_film_margin_distillation.validate import SETTINGS as V71_SETTINGS
from experiments.v84_residual_invariant_skeleton.evaluate import PAIR_SETTINGS, domains
from src.data import get_records


WEIGHTS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def prepare(fold, records):
    reference, _, v104, _ = v104_probabilities(fold, records)
    domain = domains(reference["indices"], fold, records)
    v71 = np.load(
        ROOT / f"experiments/v71_gated_film_distillation/outputs/student_f{fold}.npz"
    )["probabilities"].astype(np.float64)
    v71_prediction = np.asarray(
        run(
            fold,
            (v71, reference["indices"], reference["labels"]),
            records,
            V71_SETTINGS,
        )["predictions"],
        dtype=np.int64,
    )
    adapted = {
        weight: fused_probabilities(fold, reference, v104, records, weight)[0]
        for weight in WEIGHTS
    }
    return reference, domain, v71_prediction, adapted


def evaluate(fold, prepared, low_weight, high_weight, records):
    reference, domain, v71_prediction, adapted = prepared
    probabilities = adapted[low_weight].copy()
    probabilities[domain == 3] = adapted[high_weight][domain == 3]
    paired_prediction = np.asarray(
        run(
            fold,
            (probabilities, reference["indices"], reference["labels"]),
            records,
            PAIR_SETTINGS,
        )["predictions"],
        dtype=np.int64,
    )
    prediction = np.where(domain >= 2, paired_prediction, v71_prediction)
    return scores(prediction, reference, fold, records)


def main():
    records = get_records()
    prepared = [prepare(fold, records) for fold in (0, 1)]
    grid = []
    for low_weight in WEIGHTS:
        for high_weight in WEIGHTS:
            grid.append(
                {
                    "low_environment_weight": low_weight,
                    "high_environment_weight": high_weight,
                    "fold0": evaluate(
                        0, prepared[0], low_weight, high_weight, records
                    ),
                }
            )
    selected = max(
        grid,
        key=lambda item: (
            item["fold0"]["correct"],
            item["fold0"]["test_domain_weighted_accuracy"],
            -abs(item["low_environment_weight"] - 0.6)
            - abs(item["high_environment_weight"] - 0.6),
        ),
    )
    fold1 = evaluate(
        1,
        prepared[1],
        selected["low_environment_weight"],
        selected["high_environment_weight"],
        records,
    )
    correct = selected["fold0"]["correct"] + fold1["correct"]
    rows = selected["fold0"]["rows"] + fold1["rows"]
    report = {
        "protocol": "Two coarse weights selected on Fold0 and transferred once to Fold1.",
        "weights": WEIGHTS,
        "grid": grid,
        "selected": selected | {"fold1": fold1},
        "summary": {"correct": correct, "rows": rows, "accuracy": correct / rows},
        "v127": {"correct": 1205, "rows": 1357},
        "promote": correct > 1205,
    }
    output = HERE / "outputs/evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selected": report["selected"], "summary": report["summary"], "promote": report["promote"]}, indent=2))


if __name__ == "__main__":
    main()

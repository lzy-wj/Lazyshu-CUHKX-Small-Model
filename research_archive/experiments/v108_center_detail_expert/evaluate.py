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
    align,
    load,
    pair_metrics,
    raw_metrics,
)
from src.data import get_records


OBJECT_ACTIONS = frozenset(range(6, 28)) | frozenset(range(37, 40))


def main():
    records = get_records()
    folds = []
    for fold in (0, 1):
        reference, _, v104, domain = v104_probabilities(fold, records)
        detail = align(
            reference,
            load(
                HERE
                / "outputs"
                / "checkpoints"
                / f"v108_center_detail_f{fold}"
                / "validation.npz"
            ),
        )
        target_equal = v104.copy()
        target_equal[domain > 0] = 0.5 * (
            v104[domain > 0] + detail[domain > 0]
        )
        semantic_mask = (domain > 0) & np.asarray(
            [
                int(base) in OBJECT_ACTIONS or int(local) in OBJECT_ACTIONS
                for base, local in zip(v104.argmax(1), detail.argmax(1))
            ]
        )
        semantic_gate = v104.copy()
        semantic_gate[semantic_mask] = 0.5 * (
            v104[semantic_mask] + detail[semantic_mask]
        )
        variants = {
            "v104": v104,
            "v108_detail": detail,
            "v108_target_domain_equal": target_equal,
            "v108_semantic_object_gate": semantic_gate,
        }
        result = {
            "fold": fold,
            "semantic_gate_rows": int(semantic_mask.sum()),
            "variants": {},
        }
        for name, probabilities in variants.items():
            result["variants"][name] = {
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
        folds.append(result)

    summary = {}
    for name in folds[0]["variants"]:
        summary[name] = {}
        for stage in ("raw", "pair"):
            values = [fold["variants"][name][stage] for fold in folds]
            correct = sum(value["correct"] for value in values)
            rows = sum(value["rows"] for value in values)
            weighted = [
                value["test_domain_weighted_accuracy"] for value in values
            ]
            summary[name][stage] = {
                "correct": correct,
                "rows": rows,
                "accuracy": correct / rows,
                "fold_test_domain_weighted_accuracy": weighted,
                "mean_fold_test_domain_weighted_accuracy": float(np.mean(weighted)),
            }
    baseline = summary["v104"]["pair"]
    candidate = summary["v108_semantic_object_gate"]["pair"]
    report = {
        "protocol": (
            "Center-detail input is the only training change. The primary gate is "
            "fixed from semantic object-action names before validation inspection."
        ),
        "object_actions": sorted(OBJECT_ACTIONS),
        "folds": folds,
        "summary": summary,
        "promote": candidate["correct"] > baseline["correct"]
        and all(
            new >= old
            for new, old in zip(
                candidate["fold_test_domain_weighted_accuracy"],
                baseline["fold_test_domain_weighted_accuracy"],
            )
        ),
    }
    output = HERE / "outputs" / "evaluation.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"summary": summary, "promote": report["promote"]}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.v84_residual_invariant_skeleton.evaluate import (
    align,
    domains,
    load,
    pair_metrics,
    raw_metrics,
)
from experiments.v104_domain_gated_adapters.evaluate import fold_probabilities
from src.data import get_records


def isolated_probabilities(reference, fold, teacher):
    archive = load(
        HERE
        / "outputs"
        / f"v106_isolated_{teacher}_f{fold}"
        / "validation.npz"
    )
    positions = {int(index): position for position, index in enumerate(reference["indices"])}
    output = np.zeros_like(reference["probabilities"], dtype=np.float64)
    mask = np.zeros(len(reference["indices"]), dtype=bool)
    for source_position, index in enumerate(archive["indices"]):
        target_position = positions[int(index)]
        output[target_position] = archive["probabilities"][source_position]
        mask[target_position] = True
    return output, mask


def summarize(folds):
    summary = {}
    for name in folds[0]["variants"]:
        values = [fold["variants"][name] for fold in folds]
        summary[name] = {}
        for stage in ("raw", "pair"):
            correct = sum(value[stage]["correct"] for value in values)
            rows = sum(value[stage]["rows"] for value in values)
            weighted = [
                value[stage]["test_domain_weighted_accuracy"] for value in values
            ]
            summary[name][stage] = {
                "correct": correct,
                "rows": rows,
                "accuracy": correct / rows,
                "fold_test_domain_weighted_accuracy": weighted,
                "mean_fold_test_domain_weighted_accuracy": float(np.mean(weighted)),
            }
    return summary


def main():
    records = get_records()
    folds = []
    for fold in (0, 1):
        reference, v89, protected, old_all, refreshed_all = fold_probabilities(fold)
        domain = domains(reference["indices"], fold, records)
        old_isolated, old_mask = isolated_probabilities(reference, fold, "old")
        refreshed_isolated, refreshed_mask = isolated_probabilities(
            reference, fold, "refreshed"
        )
        expected = domain == 1
        if not np.array_equal(old_mask, expected) or not np.array_equal(
            refreshed_mask, expected
        ):
            raise RuntimeError(f"Fold {fold} isolated archives do not match domain 1")

        v104 = v89.copy()
        v104[expected] = (
            v89[expected]
            + protected[expected]
            + old_all[expected]
            + refreshed_all[expected]
        ) / 4
        v104[domain == 2] = 0.5 * (v89[domain == 2] + protected[domain == 2])

        candidate = v89.copy()
        candidate[expected] = (
            v89[expected]
            + protected[expected]
            + old_isolated[expected]
            + refreshed_isolated[expected]
        ) / 4
        candidate[domain == 2] = 0.5 * (
            v89[domain == 2] + protected[domain == 2]
        )

        variants = {
            "v89": v89,
            "v104_all_domain_adapters": v104,
            "v106_isolated_domain_adapters": candidate,
        }
        result = {"fold": fold, "variants": {}}
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

    summary = summarize(folds)
    incumbent = summary["v104_all_domain_adapters"]["pair"]
    candidate = summary["v106_isolated_domain_adapters"]["pair"]
    report = {
        "protocol": (
            "Replace only V104's domain-1 target adapters with adapters trained and "
            "normalized on domain 1; retain every route and equal fusion weight."
        ),
        "routing": {
            "domain_1": "equal V89, V97 dual, V106 isolated old, V106 isolated refreshed",
            "domain_2": "equal V89 and V97 dual",
            "domains_0_3": "V89",
        },
        "folds": folds,
        "summary": summary,
        "beats_v104": candidate["correct"] > incumbent["correct"],
        "promote": candidate["correct"] > incumbent["correct"]
        and all(
            new >= old
            for new, old in zip(
                candidate["fold_test_domain_weighted_accuracy"],
                incumbent["fold_test_domain_weighted_accuracy"],
            )
        ),
    }
    output = HERE / "outputs" / "evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"summary": summary, "promote": report["promote"]}, indent=2))


if __name__ == "__main__":
    main()

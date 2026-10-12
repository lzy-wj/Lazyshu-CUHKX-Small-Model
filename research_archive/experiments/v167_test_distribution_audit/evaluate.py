#!/usr/bin/env python3
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.v125_domain_complement.evaluate import scores
from experiments.v160_radar_batch_decoder.radar_batch import date_for_record
from experiments.v164_evidence_gated_radar.evaluate import accept
from experiments.v165_balanced_cycle_radar.evaluate import cycle_mask
from experiments.v166_repeated_transition_consensus.evaluate import (
    TEST_GROUP_BY_DATE,
    fold_inputs,
    repeated_transition_mask,
)
from experiments.v84_residual_invariant_skeleton.evaluate import domains
from src.data import get_records


GROUP_DOMAIN = {
    "unmatched": 1,
    "source20": 2,
    "source21": 2,
    "source16": 3,
}


def modality_pattern(record):
    return tuple(sorted(record["modalities"]))


def v166_prediction(fold, records):
    reference, evidence, incumbent, challenger = fold_inputs(fold, records)
    users = [records[int(index)]["user"] for index in reference["indices"]]
    selected = cycle_mask(
        reference["indices"], incumbent, challenger, records
    )
    selected |= accept(incumbent, challenger, evidence)
    selected |= repeated_transition_mask(users, incumbent, challenger, 2)
    prediction = incumbent.copy()
    prediction[selected] = challenger[selected]
    return reference, prediction


def test_pattern_counts(records):
    counts = {domain: Counter() for domain in (1, 2, 3)}
    for record in records:
        if record["split"] != "test":
            continue
        group = TEST_GROUP_BY_DATE[date_for_record(record)]
        counts[GROUP_DOMAIN[group]][modality_pattern(record)] += 1
    return counts


def matched_fold_report(fold, records, test_counts):
    reference, prediction = v166_prediction(fold, records)
    labels = reference["labels"].astype(np.int64)
    indices = reference["indices"].astype(np.int64)
    validation_domains = domains(indices, fold, records)
    patterns = np.asarray(
        [modality_pattern(records[int(index)]) for index in indices], dtype=object
    )
    domain_reports = {}
    expected_correct = 0.0
    for domain in (1, 2, 3):
        domain_rows = validation_domains == domain
        fallback_accuracy = float(
            (prediction[domain_rows] == labels[domain_rows]).mean()
        )
        pattern_reports = []
        domain_expected = 0.0
        domain_test_rows = 0
        for pattern, test_rows in sorted(test_counts[domain].items()):
            pattern_rows = domain_rows & np.asarray(
                [value == pattern for value in patterns], dtype=bool
            )
            validation_rows = int(pattern_rows.sum())
            accuracy = (
                float((prediction[pattern_rows] == labels[pattern_rows]).mean())
                if validation_rows
                else fallback_accuracy
            )
            domain_expected += test_rows * accuracy
            domain_test_rows += test_rows
            pattern_reports.append(
                {
                    "modalities": list(pattern),
                    "test_rows": test_rows,
                    "validation_rows": validation_rows,
                    "validation_accuracy": accuracy,
                    "used_domain_fallback": validation_rows == 0,
                }
            )
        expected_correct += domain_expected
        domain_reports[str(domain)] = {
            "test_rows": domain_test_rows,
            "expected_accuracy": domain_expected / domain_test_rows,
            "patterns": pattern_reports,
        }
    original = scores(prediction, reference, fold, records)
    return {
        "fold": fold,
        "original_domain_weighted_accuracy": original[
            "test_domain_weighted_accuracy"
        ],
        "modality_matched_accuracy": expected_correct / 405,
        "expected_correct_of_405": expected_correct,
        "domains": domain_reports,
    }


def main():
    records = get_records()
    test_counts = test_pattern_counts(records)
    folds = [
        matched_fold_report(fold, records, test_counts) for fold in (0, 1)
    ]
    estimate = float(
        np.mean([fold["modality_matched_accuracy"] for fold in folds])
    )
    report = {
        "protocol": (
            "Evaluate frozen V166 by test domain and exact modality-availability "
            "frequencies. Validation labels never affect test grouping or weights."
        ),
        "folds": folds,
        "mean_modality_matched_accuracy": estimate,
        "mean_expected_correct_of_405": 405 * estimate,
        "correct_needed_for_95_of_405": 0.95 * 405 - 405 * estimate,
        "reaches_95": estimate >= 0.95,
    }
    output = HERE / "outputs/evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

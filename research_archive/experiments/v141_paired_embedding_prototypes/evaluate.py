#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.research_repeat_recovery_decoder import run
from experiments.v125_domain_complement.evaluate import scores
from experiments.v135_end_to_end_label_experts.evaluate import components
from experiments.v70_film_margin_distillation.validate import SETTINGS as V71_SETTINGS
from experiments.v84_residual_invariant_skeleton.evaluate import PAIR_SETTINGS
from src.data import get_records


PAIRINGS = ({8: 18, 18: 8}, {7: 17, 17: 7})
FEATURES = {
    "fused": ("fused",),
    "visual": ("depth", "thermal", "ir"),
    "structured": ("skeleton", "imu"),
    "all": ("fused", "depth", "thermal", "ir", "skeleton", "imu"),
}
VIEWS = ("center", "mean")
RULES = ("prototype", "nearest")
STRENGTHS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


def normalize(values):
    return values / np.clip(np.linalg.norm(values, axis=1, keepdims=True), 1e-8, None)


def representation(archive, names, view):
    branches = []
    for name in names:
        center = archive[f"{name}_center"].astype(np.float64)
        values = center
        if view == "mean":
            flipped = archive[f"{name}_flip"].astype(np.float64)
            values = normalize(center + flipped)
        branches.append(normalize(values))
    return normalize(np.concatenate(branches, axis=1))


def paired_scores(archive, fold, records, feature, view, rule):
    indices = archive["indices"].astype(np.int64)
    labels = archive["labels"].astype(np.int64)
    users = np.asarray([records[int(index)]["user"] for index in indices])
    embedding = representation(archive, FEATURES[feature], view)
    output = np.zeros((len(indices), 40), dtype=np.float64)
    available = np.zeros((len(indices), 40), dtype=bool)
    for target, source in PAIRINGS[fold].items():
        source_rows = np.flatnonzero(users == source)
        target_rows = np.flatnonzero(users == target)
        for label in np.unique(labels[source_rows]):
            class_rows = source_rows[labels[source_rows] == label]
            if rule == "prototype":
                prototype = normalize(embedding[class_rows].mean(axis=0, keepdims=True))[0]
                output[target_rows, label] = embedding[target_rows] @ prototype
            else:
                output[target_rows, label] = (
                    embedding[target_rows] @ embedding[class_rows].T
                ).max(axis=1)
            available[target_rows, label] = True
    output[~available] = -1.0
    return output, available


def decode(fold, reference, v71, paired, records, v71_prediction=None):
    if v71_prediction is None:
        v71_prediction = np.asarray(
            run(
                fold,
                (v71, reference["indices"], reference["labels"]),
                records,
                V71_SETTINGS,
            )["predictions"],
            dtype=np.int64,
        )
    paired_prediction = np.asarray(
        run(
            fold,
            (paired, reference["indices"], reference["labels"]),
            records,
            PAIR_SETTINGS,
        )["predictions"],
        dtype=np.int64,
    )
    users = np.asarray([records[int(index)]["user"] for index in reference["indices"]])
    paired_users = set(PAIRINGS[fold])
    use_paired = np.asarray([user in paired_users for user in users])
    return np.where(use_paired, paired_prediction, v71_prediction)


def candidate(reference, baseline, similarity, available, strength):
    logits = np.log(np.clip(baseline, 1e-12, 1.0))
    evidence = similarity.copy()
    evidence[~available] = -1.0
    evidence = (evidence + 1.0) / 2.0
    logits = logits + strength * evidence
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def fold_values(fold, records):
    reference, v71, v114, _ = components(fold, records)
    archive = np.load(HERE / f"outputs/features_f{fold}.npz")
    if not np.array_equal(archive["indices"], reference["indices"]):
        raise ValueError("Embedding and reference indices do not align")
    return reference, v71, v114, archive


def main():
    records = get_records()
    folds = {fold: fold_values(fold, records) for fold in (0, 1)}
    reference0, v710, v1140, archive0 = folds[0]
    control0 = decode(0, reference0, v710, v1140, records)
    fixed_v710 = np.asarray(
        run(
            0,
            (v710, reference0["indices"], reference0["labels"]),
            records,
            V71_SETTINGS,
        )["predictions"],
        dtype=np.int64,
    )
    grid = [
        {
            "feature": "control",
            "view": "none",
            "rule": "none",
            "strength": 0.0,
            "fold0": scores(control0, reference0, 0, records),
        }
    ]
    diagnostics = {}
    for feature in FEATURES:
        for view in VIEWS:
            for rule in RULES:
                similarity, available = paired_scores(
                    archive0, 0, records, feature, view, rule
                )
                direct = similarity.argmax(axis=1)
                users = np.asarray(
                    [records[int(index)]["user"] for index in reference0["indices"]]
                )
                selected_rows = np.asarray([user in PAIRINGS[0] for user in users])
                diagnostics[f"{feature}_{view}_{rule}"] = {
                    "correct": int(
                        (direct[selected_rows] == reference0["labels"][selected_rows]).sum()
                    ),
                    "rows": int(selected_rows.sum()),
                }
                for strength in STRENGTHS:
                    probabilities = candidate(
                        reference0, v1140, similarity, available, strength
                    )
                    prediction = decode(
                        0,
                        reference0,
                        v710,
                        probabilities,
                        records,
                        v71_prediction=fixed_v710,
                    )
                    grid.append(
                        {
                            "feature": feature,
                            "view": view,
                            "rule": rule,
                            "strength": strength,
                            "fold0": scores(prediction, reference0, 0, records),
                        }
                    )
    selected = max(
        grid,
        key=lambda item: (
            item["fold0"]["correct"],
            item["fold0"]["test_domain_weighted_accuracy"],
            -item["strength"],
        ),
    )
    reference1, v711, v1141, archive1 = folds[1]
    if selected["feature"] == "control":
        probabilities1 = v1141
    else:
        similarity1, available1 = paired_scores(
            archive1,
            1,
            records,
            selected["feature"],
            selected["view"],
            selected["rule"],
        )
        probabilities1 = candidate(
            reference1, v1141, similarity1, available1, selected["strength"]
        )
    prediction1 = decode(1, reference1, v711, probabilities1, records)
    fold1 = scores(prediction1, reference1, 1, records)
    correct = selected["fold0"]["correct"] + fold1["correct"]
    rows = selected["fold0"]["rows"] + fold1["rows"]
    report = {
        "protocol": (
            "Fold0 selects frozen feature family, view pooling, prototype rule, "
            "and coarse strength; Fold1 is evaluated once with the fixed choice."
        ),
        "diagnostics_fold0": diagnostics,
        "grid": grid,
        "selected": selected | {"fold1": fold1},
        "summary": {"correct": correct, "rows": rows, "accuracy": correct / rows},
        "v127": {"correct": 1205, "rows": 1357},
        "promote": correct > 1205,
    }
    output = HERE / "outputs/evaluation.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "diagnostics_fold0": diagnostics,
                "selected": report["selected"],
                "summary": report["summary"],
                "promote": report["promote"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

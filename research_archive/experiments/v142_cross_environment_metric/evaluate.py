#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.v125_domain_complement.evaluate import scores
from experiments.scene_context_v13.research_repeat_recovery_decoder import run
from experiments.v135_end_to_end_label_experts.evaluate import components
from experiments.v141_paired_embedding_prototypes.evaluate import decode
from experiments.v70_film_margin_distillation.validate import SETTINGS as V71_SETTINGS
from experiments.v142_cross_environment_metric.train_metric import (
    FEATURES,
    ResidualMetric,
    normalize,
    representation,
)
from src.data import get_records


PAIRINGS = ({8: 18, 18: 8}, {7: 17, 17: 7})
RULES = ("prototype", "nearest")
STRENGTHS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


def projected_features(fold, feature):
    archive = np.load(HERE / f"outputs/features_f{fold}.npz")
    values = representation(archive, FEATURES[feature])
    checkpoint = torch.load(
        HERE / f"outputs/metric_{feature}_f{fold}.pt",
        map_location="cpu",
        weights_only=False,
    )
    model = ResidualMetric(values.shape[1], int(checkpoint["config"]["rank"]))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with torch.no_grad():
        projected = model(torch.from_numpy(values)).numpy()
    return archive, projected


def metric_scores(reference, archive, projected, fold, records, rule):
    archive_indices = archive["indices"].astype(np.int64)
    archive_labels = archive["labels"].astype(np.int64)
    archive_users = np.asarray(
        [records[int(index)]["user"] for index in archive_indices]
    )
    reference_users = np.asarray(
        [records[int(index)]["user"] for index in reference["indices"]]
    )
    output = np.full((len(reference["indices"]), 40), -1.0, dtype=np.float64)
    available = np.zeros_like(output, dtype=bool)
    index_position = {
        int(index): position for position, index in enumerate(archive_indices)
    }
    reference_features = np.stack(
        [projected[index_position[int(index)]] for index in reference["indices"]]
    )
    for target, source in PAIRINGS[fold].items():
        source_rows = np.flatnonzero(archive_users == source)
        target_rows = np.flatnonzero(reference_users == target)
        for label in np.unique(archive_labels[source_rows]):
            class_rows = source_rows[archive_labels[source_rows] == label]
            if rule == "prototype":
                prototype = normalize(
                    projected[class_rows].mean(axis=0, keepdims=True)
                )[0]
                output[target_rows, label] = reference_features[target_rows] @ prototype
            else:
                output[target_rows, label] = (
                    reference_features[target_rows] @ projected[class_rows].T
                ).max(axis=1)
            available[target_rows, label] = True
    return output, available


def add_evidence(baseline, similarity, available, strength):
    logits = np.log(np.clip(baseline, 1e-12, 1.0))
    evidence = (similarity + 1.0) / 2.0
    evidence[~available] = 0.0
    logits += strength * evidence
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def fold_components(fold, records):
    reference, v71, v114, _ = components(fold, records)
    return reference, v71, v114


def main():
    records = get_records()
    fold0 = fold_components(0, records)
    reference0, v710, v1140 = fold0
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
            "rule": "none",
            "strength": 0.0,
            "fold0": scores(control0, reference0, 0, records),
        }
    ]
    direct = {}
    cache0 = {}
    for feature in FEATURES:
        archive, projected = projected_features(0, feature)
        for rule in RULES:
            similarity, available = metric_scores(
                reference0, archive, projected, 0, records, rule
            )
            cache0[(feature, rule)] = (similarity, available)
            selected_rows = available.any(axis=1)
            direct[f"{feature}_{rule}"] = {
                "correct": int(
                    (
                        similarity[selected_rows].argmax(axis=1)
                        == reference0["labels"][selected_rows]
                    ).sum()
                ),
                "rows": int(selected_rows.sum()),
            }
            for strength in STRENGTHS:
                probabilities = add_evidence(
                    v1140, similarity, available, strength
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
    reference1, v711, v1141 = fold_components(1, records)
    if selected["feature"] == "control":
        probabilities1 = v1141
    else:
        archive1, projected1 = projected_features(1, selected["feature"])
        similarity1, available1 = metric_scores(
            reference1,
            archive1,
            projected1,
            1,
            records,
            selected["rule"],
        )
        probabilities1 = add_evidence(
            v1141, similarity1, available1, selected["strength"]
        )
    prediction1 = decode(1, reference1, v711, probabilities1, records)
    fold1_score = scores(prediction1, reference1, 1, records)
    correct = selected["fold0"]["correct"] + fold1_score["correct"]
    rows = selected["fold0"]["rows"] + fold1_score["rows"]
    report = {
        "protocol": (
            "Metric training excludes both held-out paired users. Fold0 selects "
            "representation, retrieval rule, and strength; Fold1 is blind."
        ),
        "direct_fold0": direct,
        "grid": grid,
        "selected": selected | {"fold1": fold1_score},
        "summary": {"correct": correct, "rows": rows, "accuracy": correct / rows},
        "v127": {"correct": 1205, "rows": 1357},
        "promote": correct > 1205,
    }
    output = HERE / "outputs/evaluation.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "direct_fold0": direct,
                "selected": report["selected"],
                "summary": report["summary"],
                "promote": report["promote"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

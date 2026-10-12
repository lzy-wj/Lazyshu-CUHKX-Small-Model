#!/usr/bin/env python3
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data import get_records
from src.scratch_protocol import modality_presence


MODALITIES = ["scene_depth", "thermal", "scene_ir", "skeleton", "imu", "radar"]


def class_names(records):
    names = {}
    for record in records:
        if record["split"] != "train" or record["label"] in names:
            continue
        path = next(iter(record["modalities"].values()))
        folder = next(part for part in Path(path).parts if part.startswith(f'{record["label"]}_'))
        names[record["label"]] = folder.split("_", 1)[1]
    return names


def metrics(rows, labels, predictions):
    return {
        "rows": int(rows.sum()),
        "correct": int((labels[rows] == predictions[rows]).sum()),
        "accuracy": float((labels[rows] == predictions[rows]).mean()) if rows.any() else None,
    }


def main():
    records = get_records()
    names = class_names(records)
    fold_files = [
        ROOT / f"experiments/scene_context_v13/outputs/predictions/v13_pairaware_validation_f{fold}.npz"
        for fold in (0, 1)
    ]
    parts = [np.load(path) for path in fold_files]
    indices = np.concatenate([part["indices"] for part in parts]).astype(np.int64)
    labels = np.concatenate([part["labels"] for part in parts]).astype(np.int64)
    predictions = np.concatenate([part["pseudo_labels"] for part in parts]).astype(np.int64)
    confidence = np.concatenate([part["confidence"] for part in parts])
    paired = np.concatenate([part["paired"] for part in parts]).astype(bool)
    users = np.asarray([records[index]["user"] for index in indices])
    presence = modality_presence(MODALITIES)[indices]
    all_six = presence.all(axis=1)
    visual_complete = presence[:, :3].all(axis=1)
    sensor_complete = presence[:, 3:].all(axis=1)
    report = {
        "protocol": "V13 frozen pair-aware predictions on two subject-disjoint development folds.",
        "overall": metrics(np.ones(len(indices), dtype=bool), labels, predictions),
        "paired": metrics(paired, labels, predictions),
        "unpaired": metrics(~paired, labels, predictions),
        "all_six": metrics(all_six, labels, predictions),
        "missing_any": metrics(~all_six, labels, predictions),
        "visual_complete": metrics(visual_complete, labels, predictions),
        "visual_missing": metrics(~visual_complete, labels, predictions),
        "sensor_complete": metrics(sensor_complete, labels, predictions),
        "sensor_missing": metrics(~sensor_complete, labels, predictions),
        "per_modality": {
            name: {
                "available": metrics(presence[:, position], labels, predictions),
                "missing": metrics(~presence[:, position], labels, predictions),
            }
            for position, name in enumerate(MODALITIES)
        },
        "per_user": {
            str(user): metrics(users == user, labels, predictions) for user in np.unique(users)
        },
        "confidence_bins": {},
    }
    edges = [0.0, 0.4, 0.6, 0.8, 0.9, 1.01]
    for lower, upper in zip(edges, edges[1:]):
        rows = (confidence >= lower) & (confidence < upper)
        report["confidence_bins"][f"[{lower:.2f},{upper:.2f})"] = metrics(rows, labels, predictions)
    confusions = Counter(
        (int(expected), int(actual))
        for expected, actual, is_paired in zip(labels, predictions, paired)
        if expected != actual and not is_paired
    )
    report["top_unpaired_confusions"] = [
        {
            "truth": truth,
            "truth_name": names[truth],
            "prediction": prediction,
            "prediction_name": names[prediction],
            "count": count,
        }
        for (truth, prediction), count in confusions.most_common(20)
    ]
    destination = ROOT / "experiments/scene_context_v13/outputs/v13_error_structure.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

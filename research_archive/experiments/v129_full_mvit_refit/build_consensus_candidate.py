#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def read_submission(path):
    table = pd.read_csv(path)
    if list(table.columns) != ["path", "prediction"] or len(table) != 405:
        raise ValueError(f"Invalid submission table: {path}")
    return table


def read_predictions(path):
    payload = np.load(path)
    probabilities = payload["probabilities"]
    if probabilities.shape != (405, 40):
        raise ValueError(f"Invalid probability tensor: {path}")
    return probabilities.argmax(axis=1), probabilities.max(axis=1)


def main():
    incumbent_path = (
        ROOT
        / "experiments/v24_full_refit/outputs/submissions/"
        "v24_full_warm60_pseudo_pair_aware.csv"
    )
    medium_path = (
        ROOT
        / "experiments/v127_supervision_routed_system/candidates/"
        "v127_medium_source_coverage.csv"
    )
    model_paths = [
        HERE / "predictions/conservative_pair.npz",
        HERE / "predictions/domainadapt_pair.npz",
    ]
    incumbent = read_submission(incumbent_path)
    medium = read_submission(medium_path)
    if not incumbent["path"].equals(medium["path"]):
        raise ValueError("Submission paths do not align")

    predictions = []
    confidences = []
    for path in model_paths:
        prediction, confidence = read_predictions(path)
        predictions.append(prediction)
        confidences.append(confidence)
    predictions = np.stack(predictions)
    confidences = np.stack(confidences)

    incumbent_labels = incumbent["prediction"].to_numpy(dtype=np.int64)
    challenger_labels = medium["prediction"].to_numpy(dtype=np.int64)
    proposed = challenger_labels != incumbent_labels
    unanimous = np.all(predictions == challenger_labels[None, :], axis=0)
    accepted = proposed & unanimous

    output = incumbent.copy()
    output.loc[accepted, "prediction"] = challenger_labels[accepted]
    destination = HERE / "candidates/v129_transformer_consensus.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)

    changed_positions = np.flatnonzero(accepted)
    report = {
        "protocol": (
            "Start from public V24 and accept only V127 medium-candidate changes "
            "unanimously predicted by two independently trained full-data MViTs."
        ),
        "changes_vs_public_v24": int(accepted.sum()),
        "changed_rows": [
            {
                "position": int(position),
                "path": str(output.iloc[position]["path"]),
                "incumbent": int(incumbent_labels[position]),
                "candidate": int(challenger_labels[position]),
                "mvit_confidences": [
                    float(value) for value in confidences[:, position]
                ],
            }
            for position in changed_positions
        ],
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "submitted": False,
    }
    destination.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

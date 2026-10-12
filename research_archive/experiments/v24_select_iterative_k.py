#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", nargs=2, metavar=("K", "NPZ"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_candidate(path):
    archive = np.load(path)
    required = {"probabilities", "labels", "indices"}
    if not required.issubset(archive.files):
        raise ValueError(f"{path} must contain {sorted(required)}")
    return {name: archive[name] for name in required}


def metrics(values, records, visual_presence):
    probabilities = values["probabilities"].astype(np.float64)
    labels = values["labels"].astype(np.int64)
    indices = values["indices"].astype(np.int64)
    if probabilities.shape != (len(labels), 40) or len(indices) != len(labels):
        raise ValueError("Prediction archive shapes are inconsistent")
    predictions = probabilities.argmax(axis=1)
    correct = predictions == labels
    missing_visual = ~visual_presence[indices].all(axis=1)
    users = np.asarray([records[int(index)]["user"] for index in indices])
    clipped = np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)
    return {
        "correct": int(correct.sum()),
        "rows": len(labels),
        "accuracy": float(correct.mean()),
        "nll": float(-np.log(clipped).mean()),
        "missing_visual": {
            "correct": int(correct[missing_visual].sum()),
            "rows": int(missing_visual.sum()),
            "accuracy": float(correct[missing_visual].mean()) if missing_visual.any() else None,
        },
        "complete_visual": {
            "correct": int(correct[~missing_visual].sum()),
            "rows": int((~missing_visual).sum()),
            "accuracy": float(correct[~missing_visual].mean()) if (~missing_visual).any() else None,
        },
        "per_user": {
            str(int(user)): {
                "correct": int(correct[users == user].sum()),
                "rows": int((users == user).sum()),
                "accuracy": float(correct[users == user].mean()),
            }
            for user in np.unique(users)
        },
    }


def numeric_k(value):
    try:
        return int(value)
    except ValueError:
        return 1_000_000


def main():
    args = parse_args()
    records = json.loads((ROOT / "data/metadata/samples.json").read_text())
    visual_presence = np.load(ROOT / "data/processed/video_mask.npy")[:, :3] > 0
    candidates = []
    reference_indices = None
    reference_labels = None
    for key, raw_path in args.candidate:
        path = Path(raw_path)
        values = load_candidate(path)
        if reference_indices is None:
            reference_indices = values["indices"]
            reference_labels = values["labels"]
        elif not np.array_equal(reference_indices, values["indices"]):
            raise ValueError(f"Candidate {key} indices differ from the first candidate")
        elif not np.array_equal(reference_labels, values["labels"]):
            raise ValueError(f"Candidate {key} labels differ from the first candidate")
        result = metrics(values, records, visual_presence)
        candidates.append({"k": key, "path": str(path), "metrics": result})
    selected = min(
        candidates,
        key=lambda item: (
            -item["metrics"]["correct"],
            item["metrics"]["nll"],
            -item["metrics"]["missing_visual"]["correct"],
            numeric_k(item["k"]),
            item["k"],
        ),
    )
    report = {
        "protocol": "Fold0 selection order: correct descending, NLL ascending, missing-visual correct descending, K ascending. Fold1 remains blind.",
        "candidates": candidates,
        "selected_k": selected["k"],
        "selected_path": selected["path"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

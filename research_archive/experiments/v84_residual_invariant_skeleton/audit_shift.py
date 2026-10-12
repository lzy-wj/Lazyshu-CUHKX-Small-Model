#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def main():
    records = json.loads((ROOT / "data/metadata/samples.json").read_text())
    skeleton = np.load(ROOT / "data/processed/skeleton.npy", mmap_mode="r")
    available = np.load(ROOT / "data/processed/sensor_mask.npy", mmap_mode="r")[:, 0] > 0
    pose = skeleton[:, :, :51].reshape(-1, 64, 17, 3)
    features = {
        "root_z": np.median(pose[:, :, 0, 2], axis=1),
        "torso": np.median(np.linalg.vector_norm(pose[:, :, 8] - pose[:, :, 0], axis=-1), axis=1),
        "body_span": np.median(np.ptp(pose, axis=2).max(axis=-1), axis=1),
        "root_z_motion": np.std(pose[:, :, 0, 2], axis=1),
    }
    split_masks = {
        split: np.asarray([record["split"] == split for record in records]) & available
        for split in ("train", "test")
    }
    report = {"splits": {}, "test_median_shift_in_train_standard_deviations": {}}
    for split, mask in split_masks.items():
        report["splits"][split] = {
            "rows": int(mask.sum()),
            "quantiles_05_25_50_75_95": {
                name: np.quantile(values[mask], [0.05, 0.25, 0.5, 0.75, 0.95]).tolist()
                for name, values in features.items()
            },
        }
    for name, values in features.items():
        train_values = values[split_masks["train"]]
        test_values = values[split_masks["test"]]
        report["test_median_shift_in_train_standard_deviations"][name] = float(
            (np.median(test_values) - np.median(train_values)) / np.std(train_values)
        )
    output = HERE / "outputs/skeleton_shift_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

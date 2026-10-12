import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import ROOT, ClipDataset, get_records


def statistics(values):
    delta = np.diff(values, axis=0)
    spectrum = np.abs(np.fft.rfft(values - values.mean(0), axis=0)) ** 2
    total = spectrum[1:].sum(0).clip(1e-10)
    return np.concatenate([values.mean(0), values.std(0), values.min(0), values.max(0),
                           np.median(values, axis=0), np.quantile(values, 0.1, axis=0),
                           np.quantile(values, 0.9, axis=0), np.abs(delta).mean(0), delta.std(0),
                           np.sqrt(np.square(values).mean(0)), spectrum[1:5].sum(0) / total,
                           spectrum[5:17].sum(0) / total]).astype(np.float32)


def main():
    records = get_records()
    modalities = ["skeleton", "geometry", "imu_world", "imu"]
    dataset = ClipDataset(list(range(len(records))), modalities, canonical_pose=True)
    rows = []
    for item in tqdm(dataset, total=len(dataset), desc="Sensor tree features"):
        parts = []
        for name in modalities:
            values = statistics(item[name])
            if not item[name + "_mask"]:
                values[:] = np.nan
            parts.append(values)
        rows.append(np.concatenate(parts + [np.array([item[name + "_mask"] for name in modalities], dtype=np.float32)]))
    features = np.stack(rows)
    assert features.shape == (3441, 8776) and not np.isinf(features).any()
    path = ROOT / "data/processed/tree_features.npy"
    np.save(path, features)
    report = {"shape": list(features.shape), "modalities": modalities, "canonical_pose": True,
              "statistics": ["mean", "std", "min", "max", "median", "q10", "q90", "mean_absolute_delta",
                             "delta_std", "rms", "relative_fft_bins_1_to_4", "relative_fft_bins_5_to_16"],
              "missing": "NaN feature block plus explicit modality mask",
              "protocol": "Per-clip deterministic features; no labels, user IDs, absolute timestamps or learned global normalization. FFT bins are relative to resampled clip length, not physical Hz.",
              "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (ROOT / "data/metadata/tree_features.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

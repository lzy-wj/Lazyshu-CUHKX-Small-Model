import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_data import DEVICES
from scripts.prepare_imu_world import rotate
from src.data import ROOT, get_records


def multiply(left, right):
    scalar = left[:, :1] * right[:, :1] - (left[:, 1:] * right[:, 1:]).sum(1, keepdims=True)
    vector = left[:, :1] * right[:, 1:] + right[:, :1] * left[:, 1:] + np.cross(left[:, 1:], right[:, 1:])
    return np.concatenate([scalar, vector], axis=1)


def features(record):
    output = np.zeros((64, 95), dtype=np.float32)
    if "IMU" not in record["modalities"]:
        return output, 0
    tables = [pd.read_csv(path) for path in sorted((ROOT / record["modalities"]["IMU"]).glob("*.csv"))]
    if not tables:
        return output, 0
    table = pd.concat(tables, ignore_index=True)
    if table.empty:
        return output, 0
    timestamp = pd.to_datetime(table.iloc[:, 0], errors="coerce")
    table = table.loc[timestamp.notna()].copy()
    timestamp = timestamp[timestamp.notna()]
    if len(table) < 2:
        return output, 0
    table["elapsed"] = (timestamp - timestamp.min()).dt.total_seconds().to_numpy()
    target = np.linspace(0, table["elapsed"].max(), 64)
    devices = {}
    for device in DEVICES:
        rows = table[table.iloc[:, 1].astype(str).str.startswith(device + "(")]
        rows = rows.sort_values("elapsed").drop_duplicates("elapsed")
        if rows.empty:
            continue
        values = np.nan_to_num(rows.iloc[:, 2:18].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32),
                               nan=0, posinf=0, neginf=0)
        valid = np.linalg.norm(values[:, 12:16], axis=1) > 0.5
        if not valid.any():
            continue
        values, times = values[valid], rows["elapsed"].to_numpy()[valid]
        quaternion = values[:, 12:16].copy()
        quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)
        for position in range(1, len(quaternion)):
            if np.dot(quaternion[position - 1], quaternion[position]) < 0:
                quaternion[position] *= -1
        sample = np.concatenate([values[:, :6], quaternion], axis=1)
        sample = np.stack([np.interp(target, times, sample[:, channel]) for channel in range(10)], axis=1)
        sample[:, 6:] /= np.linalg.norm(sample[:, 6:], axis=1, keepdims=True).clip(1e-6)
        devices[device] = sample
    if "WTC" not in devices:
        return output, 0
    inverse_chest = devices["WTC"][:, 6:].copy()
    inverse_chest[:, 1:] *= -1
    for position, device in enumerate(DEVICES):
        if device not in devices:
            continue
        sample = devices[device]
        relative = multiply(inverse_chest, sample[:, 6:])
        basis = rotate(np.tile(np.eye(3), (64, 1)), np.repeat(relative, 3, axis=0)).reshape(64, 9)
        acceleration = rotate(sample[:, :3], relative)
        gyroscope = rotate(sample[:, 3:6] / 180, relative)
        dynamic = acceleration - np.median(acceleration, axis=0, keepdims=True)
        output[:, position * 18:(position + 1) * 18] = np.concatenate([basis, acceleration / 4, gyroscope, dynamic], axis=1)
        output[:, 90 + position] = 1
    return output.clip(-10, 10), 1


def main():
    records = get_records()
    with ProcessPoolExecutor(max_workers=12) as pool:
        outputs = list(tqdm(pool.map(features, records, chunksize=8), total=len(records)))
    values = np.stack([entry[0] for entry in outputs])
    mask = np.array([entry[1] for entry in outputs], dtype=np.float32)
    assert values.shape == (3441, 64, 95) and np.isfinite(values).all()
    np.save(ROOT / "data/processed/imu_relative.npy", values)
    np.save(ROOT / "data/processed/imu_relative_mask.npy", mask)
    report = {"shape": list(values.shape), "present": int(mask.sum()),
              "policy": "Chest-relative orientation matrices, acceleration and angular velocity; missing chest disables this modality. No identity or absolute time features."}
    (ROOT / "data/metadata/imu_relative.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

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


def features(record):
    output = np.zeros((64, 105), dtype=np.float32)
    if "IMU" not in record["modalities"]:
        return output, 0, 0.0
    tables = [pd.read_csv(path) for path in sorted((ROOT / record["modalities"]["IMU"]).glob("*.csv"))]
    if not tables:
        return output, 0, 0.0
    table = pd.concat(tables, ignore_index=True)
    if table.empty:
        return output, 0, 0.0
    timestamps = pd.to_datetime(table.iloc[:, 0], errors="coerce")
    table = table.loc[timestamps.notna()].copy()
    timestamps = timestamps[timestamps.notna()]
    if len(table) < 2:
        return output, 0, 0.0
    table["elapsed_seconds"] = (timestamps - timestamps.min()).dt.total_seconds().to_numpy()
    duration = float(table["elapsed_seconds"].max())
    if duration <= 0:
        return output, 0, duration
    target = np.linspace(0, duration, 64)
    for position, device in enumerate(DEVICES):
        rows = table[table.iloc[:, 1].astype(str).str.startswith(device + "(")]
        rows = rows.sort_values("elapsed_seconds").drop_duplicates("elapsed_seconds")
        if len(rows) < 2:
            continue
        times = rows["elapsed_seconds"].to_numpy()
        values = np.nan_to_num(rows.iloc[:, 2:18].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32),
                               nan=0, posinf=0, neginf=0)
        acceleration = values[:, :3]
        gyroscope = values[:, 3:6] / 180
        world_acceleration = rotate(acceleration, values[:, 12:16])
        world_gyroscope = rotate(gyroscope, values[:, 12:16])
        gravity_direction = acceleration / np.linalg.norm(acceleration, axis=1, keepdims=True).clip(0.1)
        dynamic = acceleration - np.median(acceleration, axis=0, keepdims=True)
        physical_time = np.broadcast_to([np.log1p(duration) / 3, np.log1p(np.median(np.diff(times)) * 10)],
                                        (len(rows), 2))
        channels = np.concatenate([acceleration / 4, world_acceleration / 4, gyroscope, world_gyroscope,
                                   gravity_direction, dynamic, physical_time], axis=1)
        output[:, position * 20:(position + 1) * 20] = np.stack([
            np.interp(target, times, channels[:, channel]) for channel in range(20)], axis=1)
        output[:, 100 + position] = ((target >= times[0]) & (target <= times[-1])).astype(np.float32)
    return output.clip(-10, 10), int(output[:, 100:].any()), duration


def main():
    records = get_records()
    with ProcessPoolExecutor(max_workers=12) as pool:
        result = list(tqdm(pool.map(features, records, chunksize=8), total=len(records)))
    values = np.stack([entry[0] for entry in result])
    mask = np.array([entry[1] for entry in result], dtype=np.float32)
    assert values.shape == (3441, 64, 105) and np.isfinite(values).all()
    np.save(ROOT / "data/processed/imu_timed.npy", values)
    np.save(ROOT / "data/processed/imu_timed_mask.npy", mask)
    train_duration = [entry[2] for entry, record in zip(result, records) if record["split"] == "train" and entry[1]]
    report = {"clips": len(records), "present": int(mask.sum()), "shape": list(values.shape),
              "training_duration_quantiles": np.percentile(train_duration, [0, 25, 50, 75, 100]).tolist(),
              "feature_policy": "Relative elapsed time only; no absolute timestamp, user ID, device MAC or test activity labels."}
    (ROOT / "data/metadata/imu_timed.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

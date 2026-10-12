import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_data import DEVICES, interpolate
from src.data import ROOT, get_records


def rotate(vectors, quaternion):
    quaternion = quaternion / np.linalg.norm(quaternion, axis=1, keepdims=True).clip(1e-6)
    scalar = quaternion[:, :1]
    axis = quaternion[:, 1:]
    twice_cross = 2 * np.cross(axis, vectors)
    return vectors + scalar * twice_cross + np.cross(axis, twice_cross)


def features(record):
    output = np.zeros((64, 85), dtype=np.float32)
    if "IMU" not in record["modalities"]:
        return output
    tables = [pd.read_csv(path) for path in sorted((ROOT / record["modalities"]["IMU"]).glob("*.csv"))]
    if not tables:
        return output
    table = pd.concat(tables, ignore_index=True)
    if table.empty:
        return output
    for position, device in enumerate(DEVICES):
        rows = table[table.iloc[:, 1].astype(str).str.startswith(device + "(")]
        rows = rows.sort_values(table.columns[0]).drop_duplicates(table.columns[0])
        if rows.empty:
            continue
        values = np.nan_to_num(rows.iloc[:, 2:18].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32))
        acceleration = rotate(values[:, :3], values[:, 12:16])
        gyroscope = rotate(values[:, 3:6], values[:, 12:16]) / 180
        dynamic = acceleration - np.median(acceleration, axis=0, keepdims=True)
        centered_gyro = gyroscope - np.median(gyroscope, axis=0, keepdims=True)
        magnitudes = np.stack([np.linalg.norm(acceleration, axis=1), np.linalg.norm(gyroscope, axis=1),
                               np.linalg.norm(np.diff(acceleration, axis=0, prepend=acceleration[:1]), axis=1),
                               np.ones(len(rows))], axis=1)
        output[:, position * 16:(position + 1) * 16] = interpolate(np.concatenate([
            acceleration / 4, gyroscope, dynamic, centered_gyro, magnitudes], axis=1))
        output[:, 80 + position] = 1
    return output.clip(-10, 10)


def main():
    with ProcessPoolExecutor(max_workers=12) as pool:
        records = get_records()
        values = np.stack(list(tqdm(pool.map(features, records, chunksize=8), total=len(records))))
    if not np.isfinite(values).all():
        raise RuntimeError("Nonfinite IMU features")
    np.save(ROOT / "data/processed/imu_world.npy", values)
    print(values.shape)


if __name__ == "__main__":
    main()

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


def features(record):
    official = np.zeros((64, 50), dtype=np.float32)
    magnetic = np.zeros((64, 100), dtype=np.float32)
    if "IMU" not in record["modalities"]:
        return official, magnetic
    tables = [
        pd.read_csv(path)
        for path in sorted((ROOT / record["modalities"]["IMU"]).glob("*.csv"))
    ]
    if not tables:
        return official, magnetic
    table = pd.concat(tables, ignore_index=True)
    if table.empty:
        return official, magnetic
    for position, device in enumerate(DEVICES):
        rows = table[table.iloc[:, 1].astype(str).str.startswith(device + "(")]
        rows = rows.sort_values(table.columns[0]).drop_duplicates(table.columns[0])
        if rows.empty:
            continue
        numeric = np.nan_to_num(
            rows.iloc[:, 2:18].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32),
            nan=0,
            posinf=0,
            neginf=0,
        )
        acceleration = numeric[:, :3] / 4
        gyroscope = numeric[:, 3:6] / 360
        angles = np.deg2rad(numeric[:, 6:9])
        magnetometer = numeric[:, 9:12] / 100
        quaternion = numeric[:, 12:16]
        official[:, position * 9:(position + 1) * 9] = interpolate(
            np.concatenate([acceleration, gyroscope, magnetometer], axis=1)
        )
        magnetic[:, position * 19:(position + 1) * 19] = interpolate(
            np.concatenate(
                [acceleration, gyroscope, magnetometer, np.sin(angles), np.cos(angles), quaternion],
                axis=1,
            )
        )
        official[:, 45 + position] = 1
        magnetic[:, 95 + position] = 1
    return official.clip(-10, 10), magnetic.clip(-10, 10)


def main():
    records = get_records()
    with ProcessPoolExecutor(max_workers=12) as pool:
        outputs = list(tqdm(pool.map(features, records, chunksize=8), total=len(records)))
    official = np.stack([output[0] for output in outputs])
    magnetic = np.stack([output[1] for output in outputs])
    magnetometer = np.concatenate(
        [official[:, :, position * 9 + 6:position * 9 + 9] for position in range(5)],
        axis=2,
    )
    world = np.load(ROOT / "data/processed/imu_world.npy")
    world_magnetic = np.concatenate([world, magnetometer], axis=2)
    world_official = np.concatenate([world, official], axis=2)
    if official.shape != (len(records), 64, 50) or magnetic.shape != (len(records), 64, 100):
        raise RuntimeError("Unexpected IMU feature shape")
    if world_magnetic.shape != (len(records), 64, 100):
        raise RuntimeError("Unexpected world-magnetic feature shape")
    if world_official.shape != (len(records), 64, 135):
        raise RuntimeError("Unexpected world-official feature shape")
    if (not np.isfinite(official).all() or not np.isfinite(magnetic).all()
            or not np.isfinite(world_magnetic).all() or not np.isfinite(world_official).all()):
        raise RuntimeError("Nonfinite IMU features")
    np.save(ROOT / "data/processed/imu_official.npy", official)
    np.save(ROOT / "data/processed/imu_magnetic.npy", magnetic)
    np.save(ROOT / "data/processed/imu_world_magnetic.npy", world_magnetic)
    np.save(ROOT / "data/processed/imu_world_official.npy", world_official)
    report = {
        "official_shape": list(official.shape),
        "magnetic_shape": list(magnetic.shape),
        "world_magnetic_shape": list(world_magnetic.shape),
        "world_official_shape": list(world_official.shape),
        "devices": DEVICES,
        "official_channels_per_device": ["acceleration", "gyroscope", "magnetometer"],
        "magnetic_channels_per_device": [
            "acceleration", "gyroscope", "magnetometer", "euler_sin", "euler_cos", "quaternion"
        ],
        "policy": "Linear 64-step resampling with scaled local sensor coordinates and per-device masks.",
    }
    (ROOT / "data/metadata/imu_magnetic.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

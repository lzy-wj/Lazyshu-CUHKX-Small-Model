import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_imu_world import rotate
from src.data import ROOT, get_records


def main():
    raw, world, inverse, norms = [], [], [], []
    for record in [record for record in get_records() if record["split"] == "train"][::50]:
        if "IMU" not in record["modalities"]:
            continue
        for path in (ROOT / record["modalities"]["IMU"]).glob("*.csv"):
            table = pd.read_csv(path)
            if table.empty:
                continue
            values = table.iloc[:, 2:18].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
            stable = np.isfinite(values).all(1) & (np.linalg.norm(values[:, 3:6], axis=1) < 10)
            stable &= abs(np.linalg.norm(values[:, :3], axis=1) - 1) < 0.15
            values = values[stable]
            if not len(values):
                continue
            quaternion = values[:, 12:16]
            norms.extend(np.linalg.norm(quaternion, axis=1))
            raw.extend(values[:, :3])
            world.extend(rotate(values[:, :3], quaternion))
            inverse.extend(rotate(values[:, :3], quaternion * np.array([1, -1, -1, -1])))
    result = {}
    for name, values in [("raw", raw), ("world", world), ("conjugate_control", inverse)]:
        values = np.asarray(values)
        result[name] = {"rows": len(values), "median_acceleration": np.median(values, axis=0).tolist(),
                        "median_gravity_error": float(np.median(np.linalg.norm(values - [0, 0, 1], axis=1)))}
    assert result["world"]["median_gravity_error"] < 0.1
    assert result["world"]["median_gravity_error"] < result["conjugate_control"]["median_gravity_error"]
    report = {"status": "passed", "selection": "Every 50th labeled clip, finite rows with angular speed below 10 deg/s and acceleration magnitude within 0.15g of 1g; no test data", "measurements": result,
              "quaternion_norm_quantiles": np.percentile(norms, [0, 50, 100]).tolist(),
              "caveat": "This checks quaternion direction against gravity, not magnetometer/yaw calibration or inter-device timestamp synchronization"}
    (ROOT / "outputs/study_inputs/imu_orientation_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

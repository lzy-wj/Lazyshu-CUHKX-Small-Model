import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ROOT, get_records


DEVICES = ["WTLA", "WTC", "WTRA", "WTRL", "WTLL"]


COLUMNS = [
    "加速度X(g)",
    "加速度Y(g)",
    "加速度Z(g)",
    "角速度X(°/s)",
    "角速度Y(°/s)",
    "角速度Z(°/s)",
    "磁场X(uT)",
    "磁场Y(uT)",
    "磁场Z(uT)",
]


def features(record):
    output = np.zeros((50, 45), dtype=np.float32)
    if "IMU" not in record["modalities"]:
        return output
    paths = sorted((ROOT / record["modalities"]["IMU"]).glob("*.csv"))
    if not paths:
        return output
    table = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    if table.empty:
        return output
    for position, device in enumerate(DEVICES):
        rows = table[table.iloc[:, 1].astype(str).str.startswith(device + "(")]
        rows = rows.sort_values(table.columns[0]).drop_duplicates(table.columns[0])
        if rows.empty:
            continue
        values = rows[COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        values = np.nan_to_num(values, nan=0, posinf=0, neginf=0)
        output[:, position * 9:(position + 1) * 9] = resample(values, 50, axis=0).real.astype(np.float32)
    return output


def main():
    records = get_records()
    with ProcessPoolExecutor(max_workers=8) as pool:
        values = list(tqdm(pool.map(features, records, chunksize=8), total=len(records)))
    output = np.stack(values)
    if output.shape != (len(records), 50, 45) or not np.isfinite(output).all():
        raise RuntimeError(f"Unexpected exact IMU cache: {output.shape}")
    destination = ROOT / "data/processed/imu_official_exact.npy"
    np.save(destination, output)
    report = {
        "shape": list(output.shape),
        "devices": DEVICES,
        "channels_per_device": COLUMNS,
        "resampling": "scipy.signal.resample to 50 steps",
        "scaling": "none",
        "source": "official CUHK-X SM/Code/imu preprocessing",
    }
    (ROOT / "data/metadata/imu_official_exact.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

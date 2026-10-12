#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
MODALITIES = ("Depth_Color", "IR", "Thermal")
RELIABILITY = {"Depth_Color": 0.95, "IR": 0.90, "Thermal": 1.0}


def main():
    values = {
        name: json.loads((HERE / f"outputs/{name}_windows.json").read_text())
        for name in MODALITIES
    }
    count = len(values[MODALITIES[0]]["windows"])
    windows = []
    selected = []
    qualities = []
    for index in range(count):
        candidates = []
        for modality in MODALITIES:
            window = values[modality]["windows"][index]
            quality = values[modality]["quality"][index] * RELIABILITY[modality]
            if window is not None:
                candidates.append((quality, modality, window))
        if candidates:
            quality, modality, window = max(candidates)
            if quality < 0.10:
                modality, window = "none", None
        else:
            quality, modality, window = 0.0, "none", None
        windows.append(window)
        selected.append(modality)
        qualities.append(quality)
    output = HERE / "outputs/robust_windows.json"
    output.write_text(json.dumps(windows) + "\n")
    report = {
        "rows": count,
        "selected_modalities": {
            modality: selected.count(modality) for modality in (*MODALITIES, "none")
        },
        "mean_selected_quality": float(np.mean(qualities)),
        "windows": str(output),
    }
    (HERE / "outputs/merge_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

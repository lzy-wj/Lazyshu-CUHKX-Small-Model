#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.blend_aware_v12.audit_existing import temperature_scale
from src.scratch_protocol import modality_presence


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--auxiliary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    main = np.load(args.main)
    auxiliary = np.load(args.auxiliary)
    if not np.array_equal(main["indices"], auxiliary["indices"]):
        raise ValueError("main and auxiliary test predictions are not aligned")
    presence = modality_presence(["scene_depth", "thermal", "scene_ir", "skeleton", "imu", "radar"])
    indices = main["indices"]
    selected = presence[indices, 3] & ~presence[indices, :3].all(axis=1)
    probabilities = main["probabilities"].copy()
    skeleton = temperature_scale(auxiliary["probabilities_skeleton"], 0.75)
    probabilities[selected] = 0.95 * probabilities[selected] + 0.05 * skeleton[selected]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, probabilities=probabilities.astype(np.float32), indices=indices)
    print(json.dumps({"rows": len(indices), "auxiliary_rows": int(selected.sum())}, indent=2))


if __name__ == "__main__":
    main()

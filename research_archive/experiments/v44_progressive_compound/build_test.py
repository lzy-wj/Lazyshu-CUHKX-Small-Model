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
    parser.add_argument("--progressive", type=Path, required=True)
    parser.add_argument("--x3dl", type=Path, required=True)
    parser.add_argument("--skeleton-auxiliary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def align(indices, archive, key="probabilities"):
    positions = {int(index): position for position, index in enumerate(archive["indices"])}
    return archive[key][np.asarray([positions[int(index)] for index in indices])]


def main():
    args = parse_args()
    main = np.load(args.main)
    progressive = np.load(args.progressive)
    x3dl = np.load(args.x3dl)
    skeleton_auxiliary = np.load(args.skeleton_auxiliary)
    indices = main["indices"]
    probabilities = 0.9 * main["probabilities"] + 0.1 * align(indices, progressive)
    x3dl_probabilities = temperature_scale(align(indices, x3dl), 0.75)
    low_confidence = probabilities.max(axis=1) < 0.5
    probabilities[low_confidence] = (
        0.9 * probabilities[low_confidence] + 0.1 * x3dl_probabilities[low_confidence]
    )
    presence = modality_presence(["scene_depth", "thermal", "scene_ir", "skeleton", "imu", "radar"])
    visual_missing = presence[indices, 3] & ~presence[indices, :3].all(axis=1)
    skeleton = temperature_scale(
        align(indices, skeleton_auxiliary, key="probabilities_skeleton"), 0.75
    )
    probabilities[visual_missing] = (
        0.95 * probabilities[visual_missing] + 0.05 * skeleton[visual_missing]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, probabilities=probabilities.astype(np.float32), indices=indices)
    print(
        json.dumps(
            {
                "rows": len(indices),
                "x3dl_rows": int(low_confidence.sum()),
                "skeleton_auxiliary_rows": int(visual_missing.sum()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

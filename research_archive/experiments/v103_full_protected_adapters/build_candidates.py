#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def load(path):
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v89", type=Path, required=True)
    parser.add_argument("--intermediate", type=Path, required=True)
    parser.add_argument("--dual", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    v89 = load(args.v89)
    intermediate = load(args.intermediate)
    dual = load(args.dual)
    if not np.array_equal(v89["indices"], intermediate["indices"]):
        raise ValueError("V89 and intermediate predictions do not align")
    if not np.array_equal(v89["indices"], dual["indices"]):
        raise ValueError("V89 and dual predictions do not align")
    variants = {
        "v103_v89_intermediate_equal": 0.5 * (v89["probabilities"] + intermediate["probabilities"]),
        "v103_v89_dual_equal": 0.5 * (v89["probabilities"] + dual["probabilities"]),
        "v103_v89_adapter_equal": 0.5 * v89["probabilities"] + 0.25 * (
            intermediate["probabilities"] + dual["probabilities"]
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"rows": len(v89["indices"]), "variants": {}}
    for name, probabilities in variants.items():
        path = args.output_dir / f"{name}_raw.npz"
        np.savez(path, probabilities=probabilities.astype(np.float32), indices=v89["indices"])
        report["variants"][name] = {
            "path": str(path),
            "mean_confidence": float(probabilities.max(axis=1).mean()),
            "raw_changes_vs_v89": int(
                (probabilities.argmax(axis=1) != v89["probabilities"].argmax(axis=1)).sum()
            ),
        }
    (args.output_dir / "candidate_build.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

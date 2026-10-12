#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "experiments/v160_radar_batch_decoder/candidates/v127_without_0531.csv"
PROBES = {
    "row91_back": (91, 24),
    "row127_back": (127, 24),
    "row273_back": (273, 24),
    "row363_back": (363, 13),
}


def read(path):
    table = pd.read_csv(path)
    if list(table.columns) != ["path", "prediction"] or len(table) != 405:
        raise ValueError(f"Invalid submission: {path}")
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--select", nargs="*", choices=tuple(PROBES))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    base = read(BASE)
    if args.output is not None:
        output = base.copy()
        for name in args.select or []:
            position, prediction = PROBES[name]
            output.iloc[position, output.columns.get_loc("prediction")] = prediction
        destination = args.output or ROOT / "experiments/v160_radar_batch_decoder/candidates/v161_selected.csv"
        output.to_csv(destination, index=False)
        print(json.dumps({"output": str(destination), "selected": args.select,
                          "changes_vs_base": int((output.prediction != base.prediction).sum())}, indent=2))
        return
    output_dir = ROOT / "experiments/v160_radar_batch_decoder/candidates"
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"base": str(BASE.relative_to(ROOT)), "probes": {}}
    for name, (position, prediction) in PROBES.items():
        output = base.copy()
        output.iloc[position, output.columns.get_loc("prediction")] = prediction
        destination = output_dir / f"v161_probe_{name}.csv"
        output.to_csv(destination, index=False)
        report["probes"][name] = {
            "position": position,
            "prediction": prediction,
            "path": str(destination.relative_to(ROOT)),
        }
    (output_dir / "v161_public_probes.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

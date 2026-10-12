#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FALLBACK = ROOT / "experiments/v160_radar_batch_decoder/candidates/v127_without_0531.csv"
PROBES = {
    "row81_forward": (81, 20),
    "row268_forward": (268, 20),
    "row337_forward": (337, 22),
    "row350_forward": (350, 22),
}


def read(path):
    table = pd.read_csv(path)
    if list(table.columns) != ["path", "prediction"] or len(table) != 405:
        raise ValueError(f"Invalid submission: {path}")
    return table


def base_path():
    selected = ROOT / "experiments/v160_radar_batch_decoder/candidates/v162_selected.csv"
    return selected if selected.exists() else FALLBACK


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--select", nargs="*", choices=tuple(PROBES))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    base = read(base_path())
    if args.output is not None:
        output = base.copy()
        for name in args.select or []:
            position, prediction = PROBES[name]
            output.iloc[position, output.columns.get_loc("prediction")] = prediction
        destination = args.output
        output.to_csv(destination, index=False)
        print(json.dumps({"output": str(destination), "base": str(base_path()),
                          "selected": args.select or [],
                          "changes_vs_base": int((output.prediction != base.prediction).sum())}, indent=2))
        return
    output_dir = ROOT / "experiments/v160_radar_batch_decoder/candidates"
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"base": str(base_path().relative_to(ROOT)), "probes": {}}
    for name, (position, prediction) in PROBES.items():
        output = base.copy()
        output.iloc[position, output.columns.get_loc("prediction")] = prediction
        destination = output_dir / f"v163_probe_{name}.csv"
        output.to_csv(destination, index=False)
        report["probes"][name] = {"position": position, "prediction": prediction,
                                  "path": str(destination.relative_to(ROOT))}
    (output_dir / "v163_public_probes.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

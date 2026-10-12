#!/usr/bin/env python3
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "experiments/v24_full_refit/outputs/submissions/v24_full_warm60_pseudo_pair_aware.csv"
ALTERNATE = ROOT / "experiments/v24_full_refit/outputs/submissions/v59_teacher_sequence_hybrid.csv"
OUTPUT = ROOT / "experiments/public_probes/outputs"


def read(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    base = read(BASE)
    alternate = read(ALTERNATE)
    if [row["path"] for row in base] != [row["path"] for row in alternate]:
        raise ValueError("Probe inputs are not aligned")
    probes = {"0007": 6, "0056": 55, "0108": 107}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, position in probes.items():
        if base[position]["prediction"] == alternate[position]["prediction"]:
            raise ValueError(f"No probe difference at {name}")
        rows = [row.copy() for row in base]
        rows[position]["prediction"] = alternate[position]["prediction"]
        destination = OUTPUT / f"v24_probe_{name}.csv"
        with destination.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["path", "prediction"])
            writer.writeheader()
            writer.writerows(rows)
        print(destination, base[position]["prediction"], rows[position]["prediction"])


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data import get_records


UNMATCHED_ENV1_USERS = (23, 22)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--specialized", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    records = get_records()
    base = np.load(args.base)
    specialized = np.load(args.specialized)
    if not np.array_equal(base["indices"], specialized["indices"]):
        raise ValueError("Composite inputs are not aligned")
    probabilities = base["probabilities"].copy()
    target_user = UNMATCHED_ENV1_USERS[args.fold]
    selected = np.asarray([records[int(index)]["user"] == target_user for index in base["indices"]])
    probabilities[selected] = specialized["probabilities"][selected]
    labels = base["labels"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, probabilities=probabilities, labels=labels, indices=base["indices"])
    report = {
        "fold": args.fold,
        "target_user": target_user,
        "selected_rows": int(selected.sum()),
        "accuracy": float((probabilities.argmax(1) == labels).mean()),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

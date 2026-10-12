#!/usr/bin/env python3
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.build_pair_aware_submission import SOURCE_BY_DATE
from scripts.evaluate_repeat_decoder import record_interval
from src.data import get_records


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--specialized", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    records = get_records()
    base = np.load(args.base)
    specialized = np.load(args.specialized)
    if not np.array_equal(base["indices"], specialized["indices"]):
        raise ValueError("Composite inputs are not aligned")
    indices = base["indices"]
    dates = []
    for index in indices:
        interval = record_interval(records[int(index)])
        if interval is None:
            raise ValueError(f"Test clip {index} has no acquisition timestamp")
        dates.append(datetime.fromtimestamp(interval[0]).strftime("%Y-%m-%d"))
    unmatched = np.asarray([SOURCE_BY_DATE[date] is None for date in dates])
    probabilities = base["probabilities"].copy()
    probabilities[unmatched] = specialized["probabilities"][unmatched]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, probabilities=probabilities, indices=indices)
    report = {
        "base": str(args.base),
        "specialized": str(args.specialized),
        "rows": len(indices),
        "specialized_rows": int(unmatched.sum()),
        "base_rows": int((~unmatched).sum()),
        "mean_confidence": float(probabilities.max(axis=1).mean()),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from scripts.evaluate_repeat_decoder import record_interval
from src.data import get_records


SOURCE_BY_DATE = {
    "2025-05-31": 20,
    "2025-06-01": 20,
    "2025-06-02": 21,
    "2025-06-16": 16,
    "2025-06-17": 16,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v104", type=Path, required=True)
    parser.add_argument("--source16", type=Path, required=True)
    parser.add_argument("--source20", type=Path, required=True)
    parser.add_argument("--source21", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter-weight", type=float, default=0.6)
    return parser.parse_args()


def load(path):
    archive = np.load(path)
    return {
        int(index): probability.astype(np.float64)
        for index, probability in zip(archive["indices"], archive["probabilities"])
    }


def main():
    args = parse_args()
    records = get_records()
    reference = np.load(args.v104)
    indices = reference["indices"].astype(np.int64)
    baseline = reference["probabilities"].astype(np.float64)
    adapters = {
        16: load(args.source16),
        20: load(args.source20),
        21: load(args.source21),
    }
    probabilities = baseline.copy()
    routed = {16: 0, 20: 0, 21: 0}
    for position, index in enumerate(indices):
        interval = record_interval(records[int(index)])
        if interval is None:
            raise ValueError(f"Missing timestamp for test index {index}")
        date = datetime.fromtimestamp(interval[0]).strftime("%Y-%m-%d")
        source = SOURCE_BY_DATE.get(date)
        if source is None:
            continue
        if int(index) not in adapters[source]:
            raise ValueError(f"Adapter {source} does not cover test index {index}")
        probabilities[position] = (
            (1 - args.adapter_weight) * baseline[position]
            + args.adapter_weight * adapters[source][int(index)]
        )
        routed[source] += 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, probabilities=probabilities, indices=indices)
    report = {
        "rows": len(indices),
        "adapter_weight": args.adapter_weight,
        "routed_rows": {str(key): value for key, value in routed.items()},
        "raw_changes_vs_v104": int(
            (probabilities.argmax(1) != baseline.argmax(1)).sum()
        ),
        "mean_confidence": float(probabilities.max(axis=1).mean()),
        "source_by_date": SOURCE_BY_DATE,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

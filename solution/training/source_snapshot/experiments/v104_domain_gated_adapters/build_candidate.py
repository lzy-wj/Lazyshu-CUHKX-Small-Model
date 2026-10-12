#!/usr/bin/env python3
import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_repeat_decoder import record_interval
from src.data import get_records


DOMAIN_BY_DATE = {
    "2025-05-31": 2,
    "2025-06-01": 2,
    "2025-06-02": 2,
    "2025-06-12": 1,
    "2025-06-13": 1,
    "2025-06-16": 3,
    "2025-06-17": 3,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v89", type=Path, required=True)
    parser.add_argument("--protected", type=Path, required=True)
    parser.add_argument("--old-target", type=Path, required=True)
    parser.add_argument("--refreshed-target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--routing", choices=("full", "conservative"), default="full")
    return parser.parse_args()


def load_aligned(path, indices):
    archive = np.load(path)
    positions = {int(index): position for position, index in enumerate(archive["indices"])}
    return archive["probabilities"][[positions[int(index)] for index in indices]].astype(np.float64)


def main():
    args = parse_args()
    records = get_records()
    reference = np.load(args.v89)
    indices = reference["indices"].astype(np.int64)
    v89 = reference["raw_probabilities"].astype(np.float64)
    protected = load_aligned(args.protected, indices)
    old_target = load_aligned(args.old_target, indices)
    refreshed_target = load_aligned(args.refreshed_target, indices)
    domains = []
    dates = []
    for index in indices:
        interval = record_interval(records[int(index)])
        if interval is None:
            raise ValueError(f"Missing timestamp for test index {index}")
        date = datetime.fromtimestamp(interval[0]).strftime("%Y-%m-%d")
        if date not in DOMAIN_BY_DATE:
            raise ValueError(f"Unexpected test date: {date}")
        dates.append(date)
        domains.append(DOMAIN_BY_DATE[date])
    domains = np.asarray(domains, dtype=np.int64)
    candidate = v89.copy()
    candidate[domains == 1] = (
        v89[domains == 1]
        + protected[domains == 1]
        + old_target[domains == 1]
        + refreshed_target[domains == 1]
    ) / 4
    if args.routing == "full":
        candidate[domains == 2] = 0.5 * (
            v89[domains == 2] + protected[domains == 2]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, probabilities=candidate, indices=indices)
    report = {
        "rows": len(indices),
        "domain_rows": {str(key): value for key, value in sorted(Counter(domains.tolist()).items())},
        "raw_changes_vs_v89": int((candidate.argmax(axis=1) != v89.argmax(axis=1)).sum()),
        "mean_confidence": float(candidate.max(axis=1).mean()),
        "routing_mode": args.routing,
        "routing": {
            "domain_1": "equal V89, protected, old-target, refreshed-target",
            "domain_2": "equal V89 and protected" if args.routing == "full" else "V89",
            "domain_3": "V89",
        },
        "date_domains": DOMAIN_BY_DATE,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

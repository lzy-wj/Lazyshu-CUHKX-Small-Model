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
    parser.add_argument("--adapted", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    base = np.load(args.base)
    adapted = np.load(args.adapted)
    if not np.array_equal(base["indices"], adapted["indices"]):
        raise ValueError("base and adapted test predictions are not aligned")
    report = json.loads(args.settings.read_text())
    settings = report["settings"]
    records = get_records()
    probabilities = np.empty_like(adapted["probabilities"])
    counts = {}
    for position, index in enumerate(adapted["indices"]):
        start, _ = record_interval(records[int(index)])
        date = datetime.fromtimestamp(start).strftime("%Y-%m-%d")
        if date not in SOURCE_BY_DATE:
            raise ValueError(f"Unexpected test acquisition date: {date}")
        paired = SOURCE_BY_DATE[date] is not None
        environment = int(date >= "2025-06-09")
        domain = f"{'paired' if paired else 'unmatched'}_env{environment}"
        weight = settings[domain]
        probabilities[position] = (
            (1 - weight) * base["probabilities"][position]
            + weight * adapted["probabilities"][position]
        )
        counts[domain] = counts.get(domain, 0) + 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, probabilities=probabilities, indices=adapted["indices"])
    print(json.dumps({"settings": settings, "domain_rows": counts, "rows": len(probabilities)}, indent=2))


if __name__ == "__main__":
    main()

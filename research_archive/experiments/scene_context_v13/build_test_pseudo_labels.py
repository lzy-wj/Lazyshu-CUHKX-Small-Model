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
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "outputs/predictions/v13_pair_aware.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/scene_context_v13/outputs/predictions/v14_test_pseudo_labels.npz",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    records = get_records()
    archive = np.load(args.input)
    indices = archive["indices"].astype(np.int64)
    expected = np.asarray(
        [record["index"] for record in records if record["split"] == "test"],
        dtype=np.int64,
    )
    if not np.array_equal(indices, expected):
        raise ValueError("Pseudo-label archive is not aligned with the complete test split")
    decoded = archive["probabilities"]
    raw = archive["raw_probabilities"] if "raw_probabilities" in archive else decoded
    dates = []
    for index in indices:
        interval = record_interval(records[int(index)])
        if interval is None:
            raise ValueError(f"Test clip {index} has no acquisition timestamp")
        dates.append(datetime.fromtimestamp(interval[0]).strftime("%Y-%m-%d"))
    unknown_dates = sorted(set(dates) - set(SOURCE_BY_DATE))
    if unknown_dates:
        raise ValueError(f"Unexpected test acquisition dates: {unknown_dates}")
    pseudo_labels = decoded.argmax(axis=1).astype(np.int64)
    confidence = raw.max(axis=1).astype(np.float32)
    paired = np.asarray([SOURCE_BY_DATE[date] is not None for date in dates])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        pseudo_labels=pseudo_labels,
        confidence=confidence,
        paired=paired,
        dates=np.asarray(dates),
        indices=indices,
    )
    report = {
        "input": str(args.input),
        "rows": len(indices),
        "paired_rows": int(paired.sum()),
        "unmatched_rows": int((~paired).sum()),
        "confidence_at_least_0.7": int((confidence >= 0.7).sum()),
        "selected_at_0.7": int((paired | (confidence >= 0.7)).sum()),
        "mean_confidence": float(confidence.mean()),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

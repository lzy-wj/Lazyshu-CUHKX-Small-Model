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

from experiments.blend_aware_v12.audit_existing import temperature_scale
from experiments.scene_context_v13.build_pair_aware_submission import SOURCE_BY_DATE
from experiments.scene_context_v13.evaluate_candidate_domain_pairaware import apply_setting
from scripts.evaluate_repeat_decoder import record_interval
from src.data import get_records


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        type=Path,
        default=ROOT / "outputs/predictions/final_legal_multimodal_v12_lowrank.npz",
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    records = get_records()
    base_archive = np.load(args.base)
    candidate_archive = np.load(args.candidate)
    indices = base_archive["indices"]
    if not np.array_equal(indices, candidate_archive["indices"]):
        raise ValueError("Base and candidate predictions are not aligned")
    base = (
        base_archive["raw_probabilities"]
        if "raw_probabilities" in base_archive
        else base_archive["probabilities"]
    )
    candidate = candidate_archive["probabilities"]
    settings_report = json.loads(args.settings.read_text())
    settings = settings_report["domain_settings"]
    output = base.copy()
    groups = []
    for index in indices:
        interval = record_interval(records[int(index)])
        if interval is None:
            raise ValueError(f"Test clip {index} has no acquisition timestamp")
        date = datetime.fromtimestamp(interval[0]).strftime("%Y-%m-%d")
        if date not in SOURCE_BY_DATE:
            raise ValueError(f"Unexpected test acquisition date: {date}")
        paired = SOURCE_BY_DATE[date] is not None
        environment = int(date >= "2025-06-09")
        groups.append(f"{'paired' if paired else 'unmatched'}_env{environment}")
    selected_counts = Counter()
    group_counts = Counter(groups)
    for group in sorted(set(groups)):
        rows = np.asarray([value == group for value in groups])
        output, selected = apply_setting(output, candidate, settings[group], rows)
        selected_counts[group] = int(selected.sum())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, probabilities=output, raw_probabilities=output, indices=indices)
    report = {
        "base": str(args.base),
        "candidate": str(args.candidate),
        "settings": str(args.settings),
        "settings_source_score": settings_report.get("combined"),
        "rows": len(indices),
        "group_rows": dict(group_counts),
        "selected_rows": dict(selected_counts),
        "changed_argmax": int((output.argmax(axis=1) != base.argmax(axis=1)).sum()),
        "mean_confidence": float(output.max(axis=1).mean()),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

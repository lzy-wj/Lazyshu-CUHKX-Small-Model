#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.research_sequence_decoder import selected_probabilities


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--pair-aware", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def normalize(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float32)
    probabilities = np.clip(probabilities, 1e-8, None)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def main():
    args = parse_args()
    raw_probabilities, indices, labels = selected_probabilities(args.fold)
    pair_aware = np.load(args.pair_aware)
    if not np.array_equal(indices, pair_aware["indices"]):
        raise ValueError("Raw and pair-aware teacher rows are not aligned")
    if not np.array_equal(labels, pair_aware["labels"]):
        raise ValueError("Raw and pair-aware teacher labels are not aligned")

    raw_probabilities = normalize(raw_probabilities)
    structured_probabilities = normalize(pair_aware["probabilities"])
    pseudo_labels = pair_aware["pseudo_labels"].astype(np.int64)
    if not np.array_equal(structured_probabilities.argmax(axis=1), pseudo_labels):
        raise ValueError("Structured probabilities disagree with pseudo labels")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        raw_probabilities=raw_probabilities,
        probabilities=structured_probabilities,
        pseudo_labels=pseudo_labels,
        confidence=raw_probabilities.max(axis=1).astype(np.float32),
        paired=pair_aware["paired"].astype(bool),
        indices=indices.astype(np.int64),
        labels=labels.astype(np.int64),
    )
    report = {
        "fold": args.fold,
        "rows": len(indices),
        "raw_teacher_correct_audit_only": int(
            (raw_probabilities.argmax(axis=1) == labels).sum()
        ),
        "structured_teacher_correct_audit_only": int((pseudo_labels == labels).sum()),
        "structured_changes": int(
            (raw_probabilities.argmax(axis=1) != pseudo_labels).sum()
        ),
        "mean_raw_confidence": float(raw_probabilities.max(axis=1).mean()),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()


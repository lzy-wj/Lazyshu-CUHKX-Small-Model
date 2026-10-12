#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.research_flexible_pair_decoder import evaluate, prepare_fold
from experiments.scene_context_v13.research_paired_script_decoder import PAIR_SOURCES
from experiments.scene_context_v13.research_pairaware_base_blend import PAIRED_USERS
from src.data import FOLDS, get_records


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    archive = np.load(args.archive)
    probabilities = archive["probabilities"]
    labels = archive["labels"]
    prepared = prepare_fold(
        probabilities,
        archive["indices"],
        labels,
        get_records(),
        PAIR_SOURCES[args.fold],
        FOLDS[args.fold],
    )
    decoded = evaluate(
        prepared,
        template_strength=1.8,
        gap_penalty=0.8,
        maximum_edits=5,
        allowed_users=PAIRED_USERS[args.fold],
        alignment_beam_size=40,
    )
    report = {
        "fold": args.fold,
        "raw_correct": int((probabilities.argmax(axis=1) == labels).sum()),
        "raw_rows": len(labels),
        "direct_pairaware_correct": decoded["correct"],
        "direct_pairaware_accuracy": decoded["accuracy"],
        "selected_rounds": decoded["selected_rounds"],
        "changed_predictions": decoded["changed_predictions"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

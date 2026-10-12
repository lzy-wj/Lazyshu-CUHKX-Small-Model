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
    parser.add_argument("--fold0", type=Path, required=True)
    parser.add_argument("--fold1", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def score(path, fold, records):
    archive = np.load(path)
    prepared = prepare_fold(
        archive["probabilities"],
        archive["indices"],
        archive["labels"],
        records,
        PAIR_SOURCES[fold],
        FOLDS[fold],
    )
    result = evaluate(
        prepared,
        template_strength=1.8,
        gap_penalty=0.8,
        maximum_edits=5,
        allowed_users=PAIRED_USERS[fold],
        alignment_beam_size=40,
    )
    return {
        key: result[key]
        for key in ("correct", "accuracy", "selected_rounds", "changed_predictions", "per_user")
    }


def main():
    args = parse_args()
    records = get_records()
    report = {"fold0": score(args.fold0, 0, records)}
    if args.fold1:
        report["fold1"] = score(args.fold1, 1, records)
        correct = report["fold0"]["correct"] + report["fold1"]["correct"]
        report["combined"] = {
            "correct": correct,
            "rows": 1357,
            "accuracy": correct / 1357,
        }
        report["reference_correct"] = 1168
        report["promote"] = correct > 1168
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()


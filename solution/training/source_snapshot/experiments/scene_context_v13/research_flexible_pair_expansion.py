#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.research_flexible_pair_decoder import evaluate, prepare_fold
from experiments.scene_context_v13.research_pair_identification_decoder import identify_strict_pairs
from experiments.scene_context_v13.research_paired_script_decoder import PAIR_SOURCES
from experiments.scene_context_v13.research_sequence_decoder import selected_probabilities
from src.data import FOLDS, get_records


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/scene_context_v13/outputs/flexible_pair_expansion.json",
    )
    return parser.parse_args()


def run(prepared, paired_users, settings):
    return evaluate(
        prepared,
        template_strength=settings["template_strength"],
        gap_penalty=settings["gap_penalty"],
        maximum_edits=settings["maximum_edits"],
        allowed_users=paired_users,
        alignment_beam_size=40,
    )


def main():
    args = parse_args()
    started = time.monotonic()
    records = get_records()
    folds = [selected_probabilities(fold) for fold in (0, 1)]
    prepared = [
        prepare_fold(*folds[fold], records, PAIR_SOURCES[fold], FOLDS[fold])
        for fold in (0, 1)
    ]
    paired = [identify_strict_pairs(values)[0] for values in prepared]
    candidates = []
    for maximum_edits in (5, 6, 7):
        for gap_penalty in (0.6, 0.8):
            for template_strength in (1.6, 1.8, 2.0):
                settings = {
                    "maximum_edits": maximum_edits,
                    "gap_penalty": gap_penalty,
                    "template_strength": template_strength,
                }
                result = run(prepared[0], paired[0], settings)
                candidates.append(settings | {"fold0": result})
                print(
                    f"edits={maximum_edits} gap={gap_penalty:.1f} strength={template_strength:.1f} "
                    f"fold0={result['correct']}"
                )
    selected = max(
        candidates,
        key=lambda item: (
            item["fold0"]["correct"],
            -item["fold0"]["changed_predictions"],
            -item["maximum_edits"],
            -item["template_strength"],
            -item["gap_penalty"],
        ),
    )
    settings = {
        key: selected[key]
        for key in ("maximum_edits", "gap_penalty", "template_strength")
    }
    selected["fold1"] = run(prepared[1], paired[1], settings)
    correct = selected["fold0"]["correct"] + selected["fold1"]["correct"]
    rows = len(folds[0][2]) + len(folds[1][2])
    report = {
        "protocol": (
            "The original selected boundary is expanded before inspection. Fold0 selects the fixed edit prior; "
            "Fold1 is evaluated once blind with strict label-free paired-user identification."
        ),
        "identified_users": [sorted(values) for values in paired],
        "selected": selected,
        "combined": {"correct": correct, "rows": rows, "accuracy": correct / rows},
        "reference": {"correct": 1157, "rows": rows, "accuracy": 1157 / rows},
        "candidates": candidates,
        "elapsed_seconds": time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "candidates"}, indent=2))


if __name__ == "__main__":
    main()

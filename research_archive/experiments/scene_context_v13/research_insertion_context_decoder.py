#!/usr/bin/env python3
import argparse
import json
import sys
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
        default=ROOT / "experiments/scene_context_v13/outputs/insertion_context_decoder.json",
    )
    return parser.parse_args()


def run(prepared, paired_users, strength, pooling):
    return evaluate(
        prepared,
        template_strength=1.8,
        gap_penalty=0.8,
        maximum_edits=5,
        allowed_users=paired_users,
        alignment_beam_size=40,
        insertion_context_strength=strength,
        insertion_context_pool=pooling,
    )


def main():
    args = parse_args()
    records = get_records()
    folds = [selected_probabilities(fold) for fold in (0, 1)]
    prepared = [
        prepare_fold(*folds[fold], records, PAIR_SOURCES[fold], FOLDS[fold])
        for fold in (0, 1)
    ]
    paired = [identify_strict_pairs(values)[0] for values in prepared]
    candidates = []
    for pooling in ("mean", "max"):
        for strength in (0.0, 0.1, 0.2, 0.3, 0.5, 0.8):
            result = run(prepared[0], paired[0], strength, pooling)
            candidates.append(
                {"insertion_context_pool": pooling, "insertion_context_strength": strength, "fold0": result}
            )
            print(pooling, strength, result["correct"])
    selected = max(
        candidates,
        key=lambda item: (
            item["fold0"]["correct"],
            -item["fold0"]["changed_predictions"],
            -item["insertion_context_strength"],
        ),
    )
    selected["fold1"] = run(
        prepared[1],
        paired[1],
        selected["insertion_context_strength"],
        selected["insertion_context_pool"],
    )
    rows = len(folds[0][2]) + len(folds[1][2])
    correct = selected["fold0"]["correct"] + selected["fold1"]["correct"]
    report = {
        "protocol": (
            "Insertion labels receive only fold-local training-script co-occurrence evidence. Fold0 selects "
            "pooling and strength; Fold1 is evaluated once blind."
        ),
        "selected": selected,
        "combined": {"correct": correct, "rows": rows, "accuracy": correct / rows},
        "deployed_reference": {"correct": 1161, "rows": rows, "accuracy": 1161 / rows},
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "candidates"}, indent=2))


if __name__ == "__main__":
    main()

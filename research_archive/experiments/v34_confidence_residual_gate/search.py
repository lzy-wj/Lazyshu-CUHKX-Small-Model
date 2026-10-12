#!/usr/bin/env python3
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


THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 2.0)
LOW_WEIGHTS = (0.25, 0.5, 0.75)


def load_fold(fold):
    paths = {
        "base": ROOT / f"experiments/scene_context_v13/outputs/v23_parameter_soup_warm60_f{fold}.npz",
        "adapted": ROOT / f"experiments/v24_pseudo_validation/outputs/v23_warm60_pseudo_adapt_f{fold}.npz",
        "teacher": ROOT / f"experiments/scene_context_v13/outputs/predictions/v13_pairaware_validation_f{fold}.npz",
    }
    archives = {name: np.load(path) for name, path in paths.items()}
    reference = archives["adapted"]["indices"]
    if any(not np.array_equal(reference, archive["indices"]) for archive in archives.values()):
        raise ValueError(f"Fold {fold} archives are not aligned")
    return archives


def probabilities(archives, records, fold, threshold, low_weight):
    base = archives["base"]["probabilities"]
    adapted = archives["adapted"]["probabilities"]
    confidence = archives["teacher"]["confidence"]
    weights = np.ones(len(adapted), dtype=np.float32)
    for position, index in enumerate(archives["adapted"]["indices"]):
        user = records[int(index)]["user"]
        if user > 15 and user not in PAIRED_USERS[fold] and confidence[position] < threshold:
            weights[position] = low_weight
    return ((1 - weights[:, None]) * base + weights[:, None] * adapted).astype(np.float32)


def decode(values, archive, records, fold):
    return evaluate(
        prepare_fold(
            values,
            archive["indices"],
            archive["labels"],
            records,
            PAIR_SOURCES[fold],
            FOLDS[fold],
        ),
        template_strength=1.8,
        gap_penalty=0.8,
        maximum_edits=5,
        allowed_users=PAIRED_USERS[fold],
        alignment_beam_size=40,
    )


def main():
    records = get_records()
    archives = [load_fold(fold) for fold in (0, 1)]
    candidates = []
    for threshold in THRESHOLDS:
        for low_weight in LOW_WEIGHTS:
            values = probabilities(archives[0], records, 0, threshold, low_weight)
            result = decode(values, archives[0]["adapted"], records, 0)
            candidates.append(
                {
                    "confidence_threshold": threshold,
                    "low_confidence_adapted_weight": low_weight,
                    "fold0": result,
                }
            )
            print(
                json.dumps(
                    {
                        "threshold": threshold,
                        "weight": low_weight,
                        "user23": result["per_user"]["23"]["correct"],
                        "fold0": result["correct"],
                    }
                ),
                flush=True,
            )
    selected = max(
        candidates,
        key=lambda item: (
            item["fold0"]["per_user"]["23"]["correct"],
            item["fold0"]["correct"],
            -abs(item["low_confidence_adapted_weight"] - 0.5),
            item["confidence_threshold"],
        ),
    )
    threshold = selected["confidence_threshold"]
    low_weight = selected["low_confidence_adapted_weight"]
    fold0_values = probabilities(archives[0], records, 0, threshold, low_weight)
    fold1_values = probabilities(archives[1], records, 1, threshold, low_weight)
    selected["fold1"] = decode(fold1_values, archives[1]["adapted"], records, 1)
    correct = selected["fold0"]["correct"] + selected["fold1"]["correct"]
    output_root = ROOT / "experiments/v34_confidence_residual_gate/outputs"
    output_root.mkdir(parents=True, exist_ok=True)
    for fold, values in enumerate((fold0_values, fold1_values)):
        archive = archives[fold]["adapted"]
        np.savez(
            output_root / f"v34_confidence_gate_f{fold}.npz",
            probabilities=values,
            indices=archive["indices"],
            labels=archive["labels"],
        )
    report = {
        "protocol": "Confidence gate selected on unmatched environment-1 Fold 0 rows; Fold 1 blind.",
        "selected": selected,
        "combined": {"correct": correct, "rows": 1357, "accuracy": correct / 1357},
        "reference_v32": {"correct": 1174, "rows": 1357},
        "promote": correct > 1174,
        "candidates": candidates,
    }
    (output_root / "search.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "candidates"}, indent=2))


if __name__ == "__main__":
    main()

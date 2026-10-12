#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.evaluate_candidate_domain_pairaware import pair_score
from experiments.scene_context_v13.research_pairaware_base_blend import PAIRED_USERS
from experiments.scene_context_v13.research_paired_script_decoder import PAIR_SOURCES
from experiments.scene_context_v13.research_sequence_decoder import selected_probabilities
from src.data import get_records


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold0", type=Path, required=True)
    parser.add_argument("--fold1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def softmax(values):
    values = values - values.max(axis=1, keepdims=True)
    values = np.exp(values)
    return values / values.sum(axis=1, keepdims=True)


def prototype_probabilities(path, fold, indices, records, centered, temperature):
    data = np.load(path)
    positions = {int(index): position for position, index in enumerate(data["indices"])}
    output = np.zeros((len(indices), 40), dtype=np.float64)
    available = np.zeros(len(indices), dtype=bool)
    for target_user in PAIRED_USERS[fold]:
        source_user = PAIR_SOURCES[fold][int(target_user > 15)][0]
        source_rows = data["users"] == source_user
        target_output_rows = np.asarray(
            [records[int(index)]["user"] == target_user for index in indices]
        )
        target_positions = np.asarray([positions[int(index)] for index in indices[target_output_rows]])
        source = data["features"][source_rows].astype(np.float64)
        target = data["features"][target_positions].astype(np.float64)
        if centered:
            source -= source.mean(axis=0, keepdims=True)
            target -= target.mean(axis=0, keepdims=True)
        source /= np.linalg.norm(source, axis=1, keepdims=True).clip(1e-12)
        target /= np.linalg.norm(target, axis=1, keepdims=True).clip(1e-12)
        source_labels = data["labels"][source_rows]
        class_available = np.asarray([np.any(source_labels == label) for label in range(40)])
        prototypes = np.stack(
            [
                source[source_labels == label].mean(axis=0)
                if class_available[label]
                else np.zeros(source.shape[1])
                for label in range(40)
            ]
        )
        prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True).clip(1e-12)
        logits = target @ prototypes.T / temperature
        logits[:, ~class_available] = -1e4
        output[target_output_rows] = softmax(logits)
        available[target_output_rows] = True
    return output, available


def apply(base, prototype, available, weight, threshold):
    selected = available & (base.max(axis=1) < threshold)
    output = base.copy()
    output[selected] = (1 - weight) * output[selected] + weight * prototype[selected]
    return output, selected


def main():
    args = parse_args()
    records = get_records()
    folds = [selected_probabilities(fold) for fold in (0, 1)]
    paths = (args.fold0, args.fold1)
    candidates = []
    for centered in (False, True):
        for temperature in (0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3):
            prototype, available = prototype_probabilities(
                paths[0], 0, folds[0][1], records, centered, temperature
            )
            for weight in np.arange(0.05, 0.501, 0.05):
                for threshold in (0.2, 0.3, 0.4, 0.6, 1.0):
                    probabilities, selected = apply(
                        folds[0][0], prototype, available, float(weight), threshold
                    )
                    raw = probabilities.argmax(1)
                    candidates.append(
                        {
                            "centered": centered,
                            "temperature": temperature,
                            "weight": float(weight),
                            "threshold": threshold,
                            "selected_rows": int(selected.sum()),
                            "fold0_raw_correct": int((raw == folds[0][2]).sum()),
                            "fold0_paired_raw_correct": int(
                                (
                                    raw[available]
                                    == folds[0][2][available]
                                ).sum()
                            ),
                        }
                    )
    shortlist = sorted(
        candidates,
        key=lambda item: (
            item["fold0_paired_raw_correct"],
            item["fold0_raw_correct"],
            -item["weight"],
        ),
        reverse=True,
    )[:24]
    for item in shortlist:
        prototype, available = prototype_probabilities(
            paths[0], 0, folds[0][1], records, item["centered"], item["temperature"]
        )
        probabilities, _ = apply(
            folds[0][0], prototype, available, item["weight"], item["threshold"]
        )
        item["fold0"] = pair_score(0, probabilities, *folds[0][1:], records)
    selected = max(
        shortlist,
        key=lambda item: (
            item["fold0"]["correct"],
            item["fold0_paired_raw_correct"],
            item["fold0_raw_correct"],
            -item["weight"],
        ),
    )
    prototype, available = prototype_probabilities(
        paths[1], 1, folds[1][1], records, selected["centered"], selected["temperature"]
    )
    probabilities, selected_rows = apply(
        folds[1][0], prototype, available, selected["weight"], selected["threshold"]
    )
    fold1 = pair_score(1, probabilities, *folds[1][1:], records)
    correct = selected["fold0"]["correct"] + fold1["correct"]
    report = {
        "protocol": (
            "Fold0 selects cosine-prototype normalization, temperature, blend weight and confidence gate. "
            "Each prototype uses only the labeled paired source user; target centering is label-free. Fold1 is blind."
        ),
        "selected": selected,
        "fold1_selected_rows": int(selected_rows.sum()),
        "fold1": fold1,
        "combined": {"correct": correct, "rows": 1357, "accuracy": correct / 1357},
        "baseline": {"correct": 1161, "rows": 1357, "accuracy": 1161 / 1357},
        "shortlist": shortlist,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "shortlist"}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.blend_aware_v12.audit_existing import align, candidate_pairs, load, temperature_scale
from experiments.scene_context_v13.research_clip_metadata import decoder_metrics
from experiments.scene_context_v13.research_sequence_decoder import selected_probabilities
from src.data import get_records


GROUPS = {
    "hygiene_clothing": [0, 1, 2, 3, 4, 5, 16],
    "food_drink": [6, 7, 8, 9, 10, 11, 14, 37],
    "desk_document": [17, 18, 20, 21, 22, 24],
    "media_device": [19, 23, 24, 25, 26, 27],
    "exercise_posture": list(range(28, 37)),
    "health": [7, 37, 38, 39],
}


def replace_group(base, candidate, group, weight, temperature):
    output = base.copy()
    selected = np.isin(base.argmax(axis=1), group)
    if not selected.any():
        return output, selected
    group = np.asarray(group)
    base_values = base[selected][:, group]
    candidate_values = temperature_scale(candidate[selected][:, group], temperature)
    base_conditional = base_values / base_values.sum(axis=1, keepdims=True).clip(1e-12)
    conditional = (1 - weight) * base_conditional + weight * candidate_values
    output[np.ix_(selected, group)] = conditional * base_values.sum(axis=1, keepdims=True)
    return output, selected


def main():
    records = get_records()
    folds = [selected_probabilities(fold) for fold in (0, 1)]
    references = [
        {"probabilities": folds[fold][0], "indices": folds[fold][1], "labels": folds[fold][2]}
        for fold in (0, 1)
    ]
    candidates = []
    for name, fold0_path, fold1_path, key in candidate_pairs():
        values = []
        for fold, path in enumerate((fold0_path, fold1_path)):
            current = align(references[fold], load(path), key=key)
            if current is None:
                break
            values.append(current)
        if len(values) == 2:
            candidates.append((name, [fold0_path, fold1_path], key, values))
    searches = []
    for group_name, group in GROUPS.items():
        for name, paths, key, values in candidates:
            for temperature in (0.5, 0.75, 1.0, 1.5, 2.0):
                for weight in (0.25, 0.5, 0.75, 1.0):
                    probability, selected = replace_group(
                        folds[0][0], values[0], group, weight, temperature
                    )
                    searches.append(
                        {
                            "group": group_name,
                            "labels": group,
                            "candidate": name,
                            "paths": [str(path.relative_to(ROOT)) for path in paths],
                            "key": key,
                            "temperature": temperature,
                            "weight": weight,
                            "fold0_raw_correct": int((probability.argmax(1) == folds[0][2]).sum()),
                            "fold0_selected_rows": int(selected.sum()),
                            "values": values,
                        }
                    )
    searches.sort(
        key=lambda item: (item["fold0_raw_correct"], -item["weight"]), reverse=True
    )
    for item in searches[:100]:
        probability, _ = replace_group(
            folds[0][0], item["values"][0], item["labels"], item["weight"], item["temperature"]
        )
        item["fold0"] = decoder_metrics(
            probability, folds[0][1], folds[0][2], records, 0
        )
    selected = max(
        searches[:100],
        key=lambda item: (
            item["fold0"]["decoded_correct"],
            item["fold0_raw_correct"],
            -item["weight"],
        ),
    )
    fold1_probability, fold1_rows = replace_group(
        folds[1][0],
        selected["values"][1],
        selected["labels"],
        selected["weight"],
        selected["temperature"],
    )
    selected["fold1_selected_rows"] = int(fold1_rows.sum())
    selected["fold1"] = decoder_metrics(
        fold1_probability, folds[1][1], folds[1][2], records, 1
    )
    selected.pop("values")
    combined_correct = selected["fold0"]["decoded_correct"] + selected["fold1"]["decoded_correct"]
    report = {
        "protocol": (
            "Semantic groups are fixed before search. Candidate, temperature and within-group weight "
            "are selected on fold 0; fold 1 is blind. The gate uses only the baseline top-1 group."
        ),
        "selected": selected,
        "combined": {
            "correct": combined_correct,
            "rows": len(folds[0][2]) + len(folds[1][2]),
            "accuracy": combined_correct / (len(folds[0][2]) + len(folds[1][2])),
        },
        "baseline": {"correct": 1103, "accuracy": 1103 / 1357},
        "top_fold0": [
            {key: value for key, value in item.items() if key != "values"}
            for item in searches[:25]
        ],
    }
    output = ROOT / "experiments/scene_context_v13/outputs/confusion_specialists.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "top_fold0"}, indent=2))


if __name__ == "__main__":
    main()

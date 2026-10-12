#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.blend_aware_v12.audit_existing import align, load, temperature_scale
from experiments.scene_context_v13.research_flexible_pair_decoder import evaluate, prepare_fold
from experiments.scene_context_v13.research_paired_script_decoder import PAIR_SOURCES
from experiments.scene_context_v13.research_sequence_decoder import selected_probabilities
from src.data import FOLDS, get_records


PAIRED_USERS = ({8, 18}, {7, 17})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--fold0", type=Path, required=True)
    parser.add_argument("--fold1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def apply_setting(base, candidate, setting):
    scaled = temperature_scale(candidate, setting["temperature"])
    output = base.copy()
    selected = (
        np.ones(len(base), dtype=bool)
        if setting["mode"] == "global"
        else base.max(axis=1) < setting["threshold"]
    )
    output[selected] = (1 - setting["weight"]) * output[selected] + setting["weight"] * scaled[selected]
    return output, selected


def pair_aware_metrics(probabilities, indices, labels, records, fold):
    prepared = prepare_fold(
        probabilities,
        indices,
        labels,
        records,
        PAIR_SOURCES[fold],
        FOLDS[fold],
    )
    return evaluate(
        prepared,
        template_strength=1.8,
        gap_penalty=0.8,
        maximum_edits=5,
        allowed_users=PAIRED_USERS[fold],
        alignment_beam_size=40,
    )


def main():
    args = parse_args()
    records = get_records()
    folds = [selected_probabilities(fold) for fold in (0, 1)]
    candidates = []
    standalone = []
    for fold, path in enumerate((args.fold0, args.fold1)):
        reference = {
            "probabilities": folds[fold][0],
            "indices": folds[fold][1],
            "labels": folds[fold][2],
        }
        values = align(reference, load(path))
        candidates.append(values)
        standalone.append(
            {
                "correct": int((values.argmax(1) == folds[fold][2]).sum()),
                "accuracy": float((values.argmax(1) == folds[fold][2]).mean()),
            }
        )
    users = [np.asarray([records[int(index)]["user"] for index in folds[fold][1]]) for fold in (0, 1)]
    paired_rows = [np.isin(users[fold], list(PAIRED_USERS[fold])) for fold in (0, 1)]
    settings = []
    for temperature in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
        for weight in np.arange(0, 0.401, 0.025):
            setting = {
                "mode": "global",
                "temperature": temperature,
                "weight": float(weight),
                "threshold": 1.0,
            }
            probabilities, selected = apply_setting(folds[0][0], candidates[0], setting)
            setting["fold0_raw_correct"] = int((probabilities.argmax(1) == folds[0][2]).sum())
            setting["fold0_pair_raw_correct"] = int(
                (probabilities.argmax(1)[paired_rows[0]] == folds[0][2][paired_rows[0]]).sum()
            )
            setting["gated_rows_fold0"] = int(selected.sum())
            settings.append(setting)
        for threshold in (0.15, 0.2, 0.25, 0.3, 0.4):
            for weight in np.arange(0.05, 0.401, 0.05):
                setting = {
                    "mode": "low_confidence",
                    "temperature": temperature,
                    "weight": float(weight),
                    "threshold": threshold,
                }
                probabilities, selected = apply_setting(folds[0][0], candidates[0], setting)
                setting["fold0_raw_correct"] = int((probabilities.argmax(1) == folds[0][2]).sum())
                setting["fold0_pair_raw_correct"] = int(
                    (probabilities.argmax(1)[paired_rows[0]] == folds[0][2][paired_rows[0]]).sum()
                )
                setting["gated_rows_fold0"] = int(selected.sum())
                settings.append(setting)
    shortlisted = {}
    for key in ("fold0_raw_correct", "fold0_pair_raw_correct"):
        for setting in sorted(settings, key=lambda item: (item[key], -item["weight"]), reverse=True)[:25]:
            identity = (setting["mode"], setting["temperature"], setting["weight"], setting["threshold"])
            shortlisted[identity] = setting
    for setting in shortlisted.values():
        probabilities, _ = apply_setting(folds[0][0], candidates[0], setting)
        setting["fold0"] = pair_aware_metrics(probabilities, folds[0][1], folds[0][2], records, 0)
    selected = max(
        shortlisted.values(),
        key=lambda item: (
            item["fold0"]["correct"],
            item["fold0_raw_correct"],
            item["fold0_pair_raw_correct"],
            -item["weight"],
        ),
    )
    fold1_probabilities, fold1_selected = apply_setting(folds[1][0], candidates[1], selected)
    selected["gated_rows_fold1"] = int(fold1_selected.sum())
    selected["fold1_raw_correct"] = int((fold1_probabilities.argmax(1) == folds[1][2]).sum())
    selected["fold1"] = pair_aware_metrics(fold1_probabilities, folds[1][1], folds[1][2], records, 1)
    correct = selected["fold0"]["correct"] + selected["fold1"]["correct"]
    rows = len(folds[0][2]) + len(folds[1][2])
    report = {
        "name": args.name,
        "protocol": (
            "Candidate blend shortlisted by fold0 raw and paired-user raw accuracy, then selected by the frozen "
            "pair-aware decoder; fold1 is evaluated once blind."
        ),
        "standalone": standalone,
        "selected": selected,
        "combined": {"correct": correct, "rows": rows, "accuracy": correct / rows},
        "pair_aware_baseline": {"correct": 1161, "rows": rows, "accuracy": 1161 / rows},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

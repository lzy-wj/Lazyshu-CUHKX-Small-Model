#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.blend_aware_v12.audit_existing import align, apply_auxiliary, load, temperature_scale
from experiments.scene_context_v13.research_domain_base_blend import group_name
from experiments.scene_context_v13.research_flexible_pair_decoder import evaluate, prepare_fold
from experiments.scene_context_v13.research_paired_script_decoder import PAIR_SOURCES
from experiments.scene_context_v13.research_pairaware_base_blend import PAIRED_USERS
from experiments.scene_context_v13.research_sequence_decoder import selected_probabilities
from src.data import FOLDS, get_records


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--fold0", type=Path, required=True)
    parser.add_argument("--fold1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--base-variant",
        choices=("v13", "no_auxiliary", "no_adapter", "compact", "minimal"),
        default="v13",
    )
    parser.add_argument("--main-fold0", type=Path)
    parser.add_argument("--main-fold1", type=Path)
    return parser.parse_args()


def base_probabilities(fold, variant, main_path=None):
    if variant == "v13" and main_path is None:
        return selected_probabilities(fold)
    reference = load(ROOT / f"outputs/study_r2plus1d/lightweight_base_int8/base_fold{fold}.npz")
    mixup_name = "v12_mixup_f0_selected_raw_int8.npz" if fold == 0 else "v12_mixup_f1_selected_raw_int8.npz"
    mixup = align(
        reference,
        load(main_path or ROOT / "experiments/blend_aware_v12/outputs/predictions" / mixup_name),
    )
    if variant in ("no_auxiliary",):
        old = align(
            reference,
            load(ROOT / f"experiments/blend_aware_v12/outputs/predictions/v12_lowrank04_old_f{fold}.npz"),
        )
        r2 = 0.875 * mixup + 0.125 * old
    else:
        r2 = mixup
    base = reference["probabilities"]
    if variant == "minimal":
        base = temperature_scale(
            align(reference, load(ROOT / f"outputs/study_r2plus1d/x3d_f{fold}_int8.npz")),
            0.9736012920941253,
        )
    probabilities = 0.4 * base + 0.6 * temperature_scale(r2, 2.0)
    if variant == "no_adapter":
        auxiliary = align(
            reference,
            load(ROOT / f"outputs/study_x3dl_soup/selected_cropblend50_fold{fold}.npz"),
        )
        probabilities = apply_auxiliary(probabilities, auxiliary)
    return probabilities, reference["indices"], reference["labels"]


def settings_grid():
    values = [{"mode": "global", "temperature": 1.0, "weight": 0.0, "threshold": 1.0}]
    for temperature in (0.5, 0.75, 1.0, 1.5, 2.0):
        for weight in np.arange(0.05, 0.401, 0.05):
            values.append(
                {
                    "mode": "global",
                    "temperature": temperature,
                    "weight": float(weight),
                    "threshold": 1.0,
                }
            )
        for threshold in (0.15, 0.2, 0.25, 0.3, 0.4):
            for weight in np.arange(0.05, 0.401, 0.05):
                values.append(
                    {
                        "mode": "low_confidence",
                        "temperature": temperature,
                        "weight": float(weight),
                        "threshold": threshold,
                    }
                )
    return values


def apply_setting(base, candidate, setting, rows):
    output = base.copy()
    selected = rows.copy()
    if setting["mode"] == "low_confidence":
        selected &= base.max(axis=1) < setting["threshold"]
    scaled = temperature_scale(candidate, setting["temperature"])
    output[selected] = (1 - setting["weight"]) * output[selected] + setting["weight"] * scaled[selected]
    return output, selected


def pair_score(fold, probabilities, indices, labels, records):
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
    folds = [
        base_probabilities(fold, args.base_variant, path)
        for fold, path in enumerate((args.main_fold0, args.main_fold1))
    ]
    candidates = []
    users = []
    group_rows = []
    for fold, path in enumerate((args.fold0, args.fold1)):
        reference = {
            "probabilities": folds[fold][0],
            "indices": folds[fold][1],
            "labels": folds[fold][2],
        }
        candidates.append(align(reference, load(path)))
        fold_users = np.asarray([records[int(index)]["user"] for index in folds[fold][1]])
        users.append(fold_users)
        group_rows.append(
            {
                group_name(user, fold): fold_users == user
                for user in FOLDS[fold]
            }
        )

    shortlist = {}
    for setting in settings_grid():
        all_rows = np.ones(len(folds[0][1]), dtype=bool)
        probabilities, _ = apply_setting(folds[0][0], candidates[0], setting, all_rows)
        raw = probabilities.argmax(axis=1)
        setting = setting | {
            "raw_by_group": {
                group: int((raw[rows] == folds[0][2][rows]).sum())
                for group, rows in group_rows[0].items()
            }
        }
        identity = tuple(setting[key] for key in ("mode", "temperature", "weight", "threshold"))
        shortlist[identity] = setting
    selected_identities = set()
    for group in group_rows[0]:
        ranked = sorted(
            shortlist.items(),
            key=lambda item: (item[1]["raw_by_group"][group], -item[1]["weight"]),
            reverse=True,
        )
        selected_identities.update(identity for identity, _ in ranked[:12])

    evaluated = []
    all_rows = np.ones(len(folds[0][1]), dtype=bool)
    for identity in selected_identities:
        setting = shortlist[identity]
        probabilities, selected_rows = apply_setting(folds[0][0], candidates[0], setting, all_rows)
        result = pair_score(0, probabilities, *folds[0][1:], records)
        evaluated.append(setting | {"selected_rows": int(selected_rows.sum()), "fold0": result})

    domain_settings = {}
    for user in FOLDS[0]:
        group = group_name(user, 0)
        selected = max(
            evaluated,
            key=lambda item: (
                item["fold0"]["per_user"][str(user)]["correct"],
                item["raw_by_group"][group],
                -item["weight"],
            ),
        )
        domain_settings[group] = {
            key: selected[key] for key in ("mode", "temperature", "weight", "threshold")
        }

    outputs = []
    selected_counts = []
    baseline_outputs = [
        pair_score(fold, folds[fold][0], *folds[fold][1:], records)
        for fold in (0, 1)
    ]
    for fold in (0, 1):
        probabilities = folds[fold][0].copy()
        counts = {}
        for group, rows in group_rows[fold].items():
            probabilities, selected_rows = apply_setting(
                probabilities,
                candidates[fold],
                domain_settings[group],
                rows,
            )
            counts[group] = int(selected_rows.sum())
        outputs.append(pair_score(fold, probabilities, *folds[fold][1:], records))
        selected_counts.append(counts)

    rows = len(folds[0][2]) + len(folds[1][2])
    correct = outputs[0]["correct"] + outputs[1]["correct"]
    report = {
        "name": args.name,
        "base_variant": args.base_variant,
        "main_paths": [str(args.main_fold0), str(args.main_fold1)],
        "protocol": (
            "Fold0 selects one regularized candidate gate for each paired-status/environment domain. The four "
            "fixed gates are transferred once to Fold1 blind under the frozen pair-aware decoder."
        ),
        "standalone": [
            {
                "correct": int((candidates[fold].argmax(1) == folds[fold][2]).sum()),
                "accuracy": float((candidates[fold].argmax(1) == folds[fold][2]).mean()),
            }
            for fold in (0, 1)
        ],
        "domain_settings": domain_settings,
        "selected_rows": selected_counts,
        "fold0": outputs[0],
        "fold1": outputs[1],
        "combined": {"correct": correct, "rows": rows, "accuracy": correct / rows},
        "pair_aware_baseline": {
            "correct": sum(result["correct"] for result in baseline_outputs),
            "rows": rows,
            "accuracy": sum(result["correct"] for result in baseline_outputs) / rows,
            "folds": baseline_outputs,
        },
        "evaluated_settings": evaluated,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "evaluated_settings"}, indent=2))


if __name__ == "__main__":
    main()

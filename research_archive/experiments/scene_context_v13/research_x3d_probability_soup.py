#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.blend_aware_v12.audit_existing import align, load, temperature_scale
from experiments.scene_context_v13.research_clip_metadata import decoder_metrics
from experiments.scene_context_v13.research_sequence_decoder import selected_probabilities
from src.data import get_records


MODEL_PATHS = {
    "old": (
        ROOT / "outputs/checkpoints/v8_x3dl_f0/validation.npz",
        ROOT / "outputs/checkpoints/v8_x3dl_f1/validation.npz",
    ),
    "sync": (
        ROOT / "outputs/checkpoints/v13_x3dl_sync_f0/validation.npz",
        ROOT / "outputs/checkpoints/v13_x3dl_sync_f1/validation.npz",
    ),
    "detail": (
        ROOT / "outputs/checkpoints/v13_x3dl_detail_f0/validation.npz",
        ROOT / "outputs/checkpoints/v13_x3dl_detail_f1/validation.npz",
    ),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/scene_context_v13/outputs/x3d_probability_soup.json",
    )
    return parser.parse_args()


def mix_models(models, weights):
    result = sum(weight * models[name] for name, weight in weights.items())
    return result / result.sum(axis=1, keepdims=True)


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


def main():
    args = parse_args()
    records = get_records()
    folds = [selected_probabilities(fold) for fold in (0, 1)]
    models = []
    for fold in (0, 1):
        reference = {"probabilities": folds[fold][0], "indices": folds[fold][1], "labels": folds[fold][2]}
        models.append({name: align(reference, load(paths[fold])) for name, paths in MODEL_PATHS.items()})
    mixtures = []
    for sync_weight in np.arange(0, 1.001, 0.25):
        for detail_weight in np.arange(0, 1.001 - sync_weight, 0.25):
            old_weight = 1 - sync_weight - detail_weight
            weights = {"old": float(old_weight), "sync": float(sync_weight), "detail": float(detail_weight)}
            mixtures.append((weights, [mix_models(models[fold], weights) for fold in (0, 1)]))
    candidates = []
    for weights, mixture in mixtures:
        for temperature in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
            scaled = temperature_scale(mixture[0], temperature)
            for weight in np.arange(0, 0.401, 0.025):
                probabilities = (1 - weight) * folds[0][0] + weight * scaled
                candidates.append(
                    {
                        "mixture": weights,
                        "mode": "global",
                        "temperature": temperature,
                        "weight": float(weight),
                        "threshold": 1.0,
                        "fold0_raw_correct": int((probabilities.argmax(1) == folds[0][2]).sum()),
                        "probabilities": probabilities,
                        "fold1_candidate": mixture[1],
                    }
                )
            for threshold in (0.15, 0.2, 0.25, 0.3, 0.4):
                selected = folds[0][0].max(axis=1) < threshold
                for weight in np.arange(0.05, 0.401, 0.05):
                    probabilities = folds[0][0].copy()
                    probabilities[selected] = (1 - weight) * probabilities[selected] + weight * scaled[selected]
                    candidates.append(
                        {
                            "mixture": weights,
                            "mode": "low_confidence",
                            "temperature": temperature,
                            "weight": float(weight),
                            "threshold": threshold,
                            "fold0_raw_correct": int((probabilities.argmax(1) == folds[0][2]).sum()),
                            "gated_rows_fold0": int(selected.sum()),
                            "probabilities": probabilities,
                            "fold1_candidate": mixture[1],
                        }
                    )
    candidates.sort(key=lambda item: item["fold0_raw_correct"], reverse=True)
    for item in candidates[:160]:
        item["fold0"] = decoder_metrics(item["probabilities"], folds[0][1], folds[0][2], records, 0)
    selected = max(
        candidates[:160],
        key=lambda item: (item["fold0"]["decoded_correct"], item["fold0_raw_correct"], -item["weight"]),
    )
    fold1_probabilities, fold1_selected = apply_setting(folds[1][0], selected["fold1_candidate"], selected)
    selected["gated_rows_fold1"] = int(fold1_selected.sum())
    selected["fold1"] = decoder_metrics(fold1_probabilities, folds[1][1], folds[1][2], records, 1)
    selected.pop("probabilities")
    selected.pop("fold1_candidate")
    correct = selected["fold0"]["decoded_correct"] + selected["fold1"]["decoded_correct"]
    rows = len(folds[0][2]) + len(folds[1][2])
    report = {
        "protocol": "X3D mixture and baseline blend selected on fold0; fold1 evaluated once blind.",
        "selected": selected,
        "combined": {"correct": correct, "rows": rows, "accuracy": correct / rows},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
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


def main():
    records = get_records()
    folds = [selected_probabilities(fold) for fold in (0, 1)]
    radar = []
    for fold in (0, 1):
        reference = {
            "probabilities": folds[fold][0],
            "indices": folds[fold][1],
            "labels": folds[fold][2],
        }
        source = load(
            ROOT
            / f"experiments/scene_context_v13/outputs/checkpoints/v13_radar_point_f{fold}/validation.npz"
        )
        positions = {int(index): position for position, index in enumerate(source["indices"])}
        order = np.asarray([positions[int(index)] for index in reference["indices"]])
        radar.append(
            {
                "probabilities": align(reference, source),
                "available": source["available"][order],
            }
        )
    candidates = []
    for temperature in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        scaled = [temperature_scale(item["probabilities"], temperature) for item in radar]
        for weight in np.arange(0, 0.201, 0.025):
            probabilities = folds[0][0].copy()
            available = radar[0]["available"].astype(bool)
            probabilities[available] = (
                (1 - weight) * probabilities[available] + weight * scaled[0][available]
            )
            candidates.append(
                {
                    "temperature": temperature,
                    "weight": float(weight),
                    "raw_correct": int((probabilities.argmax(1) == folds[0][2]).sum()),
                    "scaled": scaled,
                }
            )
    candidates.sort(key=lambda item: item["raw_correct"], reverse=True)
    for item in candidates[:20]:
        probabilities = folds[0][0].copy()
        available = radar[0]["available"].astype(bool)
        probabilities[available] = (
            (1 - item["weight"]) * probabilities[available]
            + item["weight"] * item["scaled"][0][available]
        )
        item["fold0"] = decoder_metrics(
            probabilities, folds[0][1], folds[0][2], records, 0
        )
    selected = max(
        candidates[:20],
        key=lambda item: (item["fold0"]["decoded_correct"], item["raw_correct"], -item["weight"]),
    )
    probabilities = folds[1][0].copy()
    available = radar[1]["available"].astype(bool)
    probabilities[available] = (
        (1 - selected["weight"]) * probabilities[available]
        + selected["weight"] * selected["scaled"][1][available]
    )
    selected["fold1"] = decoder_metrics(
        probabilities, folds[1][1], folds[1][2], records, 1
    )
    selected.pop("scaled")
    report = {
        "protocol": "Radar temperature and weight selected on fold0; fold1 blind; fusion only where radar exists.",
        "selected": selected,
        "combined_correct": selected["fold0"]["decoded_correct"] + selected["fold1"]["decoded_correct"],
    }
    output = ROOT / "experiments/scene_context_v13/outputs/radar_blend.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

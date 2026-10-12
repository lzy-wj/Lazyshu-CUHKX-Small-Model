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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--fold0", type=Path, required=True)
    parser.add_argument("--fold1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    records = get_records()
    folds = [selected_probabilities(fold) for fold in (0, 1)]
    candidate = []
    for fold, path in enumerate((args.fold0, args.fold1)):
        reference = {
            "probabilities": folds[fold][0],
            "indices": folds[fold][1],
            "labels": folds[fold][2],
        }
        candidate.append(align(reference, load(path)))
    standalone = [
        {
            "correct": int((candidate[fold].argmax(1) == folds[fold][2]).sum()),
            "accuracy": float((candidate[fold].argmax(1) == folds[fold][2]).mean()),
        }
        for fold in (0, 1)
    ]
    grids = []
    for temperature in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0):
        scaled = [temperature_scale(values, temperature) for values in candidate]
        for weight in np.arange(0, 0.401, 0.025):
            probabilities = (1 - weight) * folds[0][0] + weight * scaled[0]
            grids.append(
                {
                    "mode": "global",
                    "temperature": temperature,
                    "weight": float(weight),
                    "threshold": 1.0,
                    "fold0_raw_correct": int((probabilities.argmax(1) == folds[0][2]).sum()),
                    "scaled": scaled,
                }
            )
        for threshold in (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5):
            selected = folds[0][0].max(axis=1) < threshold
            for weight in np.arange(0.05, 0.401, 0.05):
                probabilities = folds[0][0].copy()
                probabilities[selected] = (
                    (1 - weight) * probabilities[selected] + weight * scaled[0][selected]
                )
                grids.append(
                    {
                        "mode": "low_confidence",
                        "temperature": temperature,
                        "weight": float(weight),
                        "threshold": threshold,
                        "fold0_raw_correct": int((probabilities.argmax(1) == folds[0][2]).sum()),
                        "gated_rows_fold0": int(selected.sum()),
                        "scaled": scaled,
                    }
                )
    grids.sort(key=lambda item: item["fold0_raw_correct"], reverse=True)
    for item in grids[:100]:
        probabilities = folds[0][0].copy()
        selected = (
            np.ones(len(probabilities), dtype=bool)
            if item["mode"] == "global"
            else probabilities.max(axis=1) < item["threshold"]
        )
        probabilities[selected] = (
            (1 - item["weight"]) * probabilities[selected]
            + item["weight"] * item["scaled"][0][selected]
        )
        item["fold0"] = decoder_metrics(
            probabilities, folds[0][1], folds[0][2], records, 0
        )
    selected = max(
        grids[:100],
        key=lambda item: (
            item["fold0"]["decoded_correct"],
            item["fold0_raw_correct"],
            -item["weight"],
        ),
    )
    probabilities = folds[1][0].copy()
    gated = (
        np.ones(len(probabilities), dtype=bool)
        if selected["mode"] == "global"
        else probabilities.max(axis=1) < selected["threshold"]
    )
    probabilities[gated] = (
        (1 - selected["weight"]) * probabilities[gated]
        + selected["weight"] * selected["scaled"][1][gated]
    )
    selected["gated_rows_fold1"] = int(gated.sum())
    selected["fold1"] = decoder_metrics(
        probabilities, folds[1][1], folds[1][2], records, 1
    )
    selected.pop("scaled")
    report = {
        "name": args.name,
        "protocol": "Temperature, global/low-confidence mode and weight selected on fold0; fold1 blind.",
        "standalone": standalone,
        "selected": selected,
        "combined": {
            "correct": selected["fold0"]["decoded_correct"] + selected["fold1"]["decoded_correct"],
            "rows": len(folds[0][2]) + len(folds[1][2]),
        },
    }
    report["combined"]["accuracy"] = report["combined"]["correct"] / report["combined"]["rows"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

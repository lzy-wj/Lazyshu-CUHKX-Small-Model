#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.blend_aware_v12.audit_existing import align, load, temperature_scale
from experiments.scene_context_v13.research_flexible_pair_decoder import evaluate, prepare_fold
from experiments.scene_context_v13.research_paired_script_decoder import PAIR_SOURCES
from experiments.scene_context_v13.research_pairaware_base_blend import PAIRED_USERS
from src.data import FOLDS, get_records
from src.scratch_protocol import modality_presence


TEMPERATURES = (0.75, 1.0, 1.5, 2.0, 3.0)
WEIGHTS = (0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25)
SCOPES = ("all", "confidence_040", "confidence_050", "confidence_060", "confidence_070", "confidence_080", "visual_missing")


def selected_rows(scope, main, indices, presence):
    if scope == "all":
        return np.ones(len(main), dtype=bool)
    if scope == "visual_missing":
        return ~presence[indices, :3].all(axis=1)
    if scope.startswith("confidence_"):
        threshold = int(scope.rsplit("_", 1)[1]) / 100
        return main.max(axis=1) < threshold
    raise ValueError(scope)


def blend(main, auxiliary, indices, presence, temperature, weight, scope):
    probabilities = main.copy()
    selected = selected_rows(scope, main, indices, presence)
    scaled = temperature_scale(auxiliary, temperature)
    probabilities[selected] = (1 - weight) * main[selected] + weight * scaled[selected]
    return probabilities


def decode(probabilities, archive, records, fold):
    return evaluate(
        prepare_fold(
            probabilities,
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
    presence = modality_presence(["scene_depth", "thermal", "scene_ir", "skeleton", "imu", "radar"])
    mains = [
        load(ROOT / f"experiments/v24_pseudo_validation/outputs/v23_warm60_pseudo_adapt_f{fold}.npz")
        for fold in (0, 1)
    ]
    auxiliaries = [
        align(
            mains[fold],
            load(ROOT / f"outputs/study_x3dl_soup/selected_cropblend50_fold{fold}.npz"),
        )
        for fold in (0, 1)
    ]
    if any(values is None for values in auxiliaries):
        raise ValueError("X3D-L validation predictions cannot be aligned")
    candidates = []
    for scope in SCOPES:
        for temperature in TEMPERATURES:
            for weight in WEIGHTS:
                probabilities = blend(
                    mains[0]["probabilities"],
                    auxiliaries[0],
                    mains[0]["indices"],
                    presence,
                    temperature,
                    weight,
                    scope,
                )
                candidates.append(
                    {
                        "scope": scope,
                        "temperature": temperature,
                        "weight": weight,
                        "fold0_raw_correct": int(
                            (probabilities.argmax(axis=1) == mains[0]["labels"]).sum()
                        ),
                    }
                )
    shortlisted = sorted(
        candidates,
        key=lambda item: (item["fold0_raw_correct"], -item["weight"]),
        reverse=True,
    )[:25]
    for item in shortlisted:
        probabilities = blend(
            mains[0]["probabilities"],
            auxiliaries[0],
            mains[0]["indices"],
            presence,
            item["temperature"],
            item["weight"],
            item["scope"],
        )
        item["fold0"] = decode(probabilities, mains[0], records, 0)
        print(json.dumps(item), flush=True)
    selected = max(
        shortlisted,
        key=lambda item: (
            item["fold0"]["correct"],
            item["fold0_raw_correct"],
            -item["weight"],
        ),
    )
    fold1_probabilities = blend(
        mains[1]["probabilities"],
        auxiliaries[1],
        mains[1]["indices"],
        presence,
        selected["temperature"],
        selected["weight"],
        selected["scope"],
    )
    selected["fold1"] = decode(fold1_probabilities, mains[1], records, 1)
    correct = selected["fold0"]["correct"] + selected["fold1"]["correct"]
    output = ROOT / "experiments/v38_x3dl_diversity/outputs/search.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol": "X3D-L settings shortlisted and selected on Fold 0; Fold 1 evaluated once blind.",
        "selected": selected,
        "combined": {"correct": correct, "rows": 1357, "accuracy": correct / 1357},
        "reference": {"correct": 1168, "rows": 1357},
        "promote": correct > 1168,
        "top_fold0": shortlisted,
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "top_fold0"}, indent=2))


if __name__ == "__main__":
    main()

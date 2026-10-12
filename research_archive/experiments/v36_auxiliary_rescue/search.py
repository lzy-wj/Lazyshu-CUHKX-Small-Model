#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.blend_aware_v12.audit_existing import temperature_scale
from experiments.scene_context_v13.research_flexible_pair_decoder import evaluate, prepare_fold
from experiments.scene_context_v13.research_paired_script_decoder import PAIR_SOURCES
from experiments.scene_context_v13.research_pairaware_base_blend import PAIRED_USERS
from src.data import FOLDS, get_records
from src.scratch_protocol import modality_presence


TEMPERATURES = (0.75, 1.0, 1.25, 1.5, 2.0)
WEIGHTS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4)
SCOPES = ("available", "visual_missing", "low_confidence")


def load_fold(fold):
    main = np.load(
        ROOT / f"experiments/v24_pseudo_validation/outputs/v23_warm60_pseudo_adapt_f{fold}.npz"
    )
    auxiliary = np.load(
        ROOT / f"experiments/v36_auxiliary_rescue/outputs/v23_aux_adabn_f{fold}.npz"
    )
    if not np.array_equal(main["indices"], auxiliary["indices"]):
        raise ValueError(f"Fold {fold} predictions are not aligned")
    return main, auxiliary


def mask_for(scope, main, indices, presence):
    skeleton_available = presence[indices, 3]
    if scope == "available":
        return skeleton_available
    if scope == "visual_missing":
        return skeleton_available & ~presence[indices, :3].all(axis=1)
    if scope == "low_confidence":
        return skeleton_available & (main.max(axis=1) < 0.6)
    raise ValueError(scope)


def blend(main, auxiliary, indices, presence, temperature, weight, scope):
    result = main.copy()
    selected = mask_for(scope, main, indices, presence)
    scaled = temperature_scale(auxiliary, temperature)
    result[selected] = (1 - weight) * main[selected] + weight * scaled[selected]
    return result


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
    folds = [load_fold(fold) for fold in (0, 1)]
    main0, auxiliary0 = folds[0]
    candidates = []
    for scope in SCOPES:
        for temperature in TEMPERATURES:
            for weight in WEIGHTS:
                probabilities = blend(
                    main0["probabilities"],
                    auxiliary0["probabilities_skeleton"],
                    main0["indices"],
                    presence,
                    temperature,
                    weight,
                    scope,
                )
                raw = int((probabilities.argmax(axis=1) == main0["labels"]).sum())
                candidates.append(
                    {
                        "scope": scope,
                        "temperature": temperature,
                        "weight": weight,
                        "fold0_raw_correct": raw,
                    }
                )
    shortlisted = sorted(
        candidates,
        key=lambda item: (item["fold0_raw_correct"], -item["weight"]),
        reverse=True,
    )[:20]
    for item in shortlisted:
        probabilities = blend(
            main0["probabilities"],
            auxiliary0["probabilities_skeleton"],
            main0["indices"],
            presence,
            item["temperature"],
            item["weight"],
            item["scope"],
        )
        item["fold0"] = decode(probabilities, main0, records, 0)
        print(json.dumps(item), flush=True)
    selected = max(
        shortlisted,
        key=lambda item: (
            item["fold0"]["correct"],
            item["fold0_raw_correct"],
            -item["weight"],
        ),
    )
    main1, auxiliary1 = folds[1]
    fold1_probabilities = blend(
        main1["probabilities"],
        auxiliary1["probabilities_skeleton"],
        main1["indices"],
        presence,
        selected["temperature"],
        selected["weight"],
        selected["scope"],
    )
    selected["fold1"] = decode(fold1_probabilities, main1, records, 1)
    correct = selected["fold0"]["correct"] + selected["fold1"]["correct"]
    report = {
        "protocol": "Auxiliary settings shortlisted and selected on Fold 0; Fold 1 evaluated once blind.",
        "selected": selected,
        "combined": {"correct": correct, "rows": 1357, "accuracy": correct / 1357},
        "reference": {"correct": 1168, "rows": 1357},
        "promote": correct > 1168,
        "top_fold0": shortlisted,
    }
    output = ROOT / "experiments/v36_auxiliary_rescue/outputs/search.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "top_fold0"}, indent=2))


if __name__ == "__main__":
    main()

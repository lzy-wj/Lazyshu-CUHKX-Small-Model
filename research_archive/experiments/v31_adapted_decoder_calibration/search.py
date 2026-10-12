#!/usr/bin/env python3
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.research_flexible_pair_decoder import evaluate, prepare_fold
from experiments.scene_context_v13.research_paired_script_decoder import PAIR_SOURCES
from experiments.scene_context_v13.research_pairaware_base_blend import PAIRED_USERS
from src.data import FOLDS, get_records


BASELINE = {
    "template_strength": 1.8,
    "gap_penalty": 0.8,
    "maximum_edits": 5,
    "minimum_margin": 0.0,
}


def score(prepared, settings):
    return evaluate(
        prepared,
        template_strength=settings["template_strength"],
        gap_penalty=settings["gap_penalty"],
        maximum_edits=settings["maximum_edits"],
        allowed_users=settings["allowed_users"],
        alignment_beam_size=40,
        minimum_margin=settings["minimum_margin"],
    )


def distance(settings):
    return (
        abs(settings["template_strength"] - BASELINE["template_strength"]) / 0.4
        + abs(settings["gap_penalty"] - BASELINE["gap_penalty"]) / 0.2
        + abs(settings["maximum_edits"] - BASELINE["maximum_edits"])
        + abs(settings["minimum_margin"] - BASELINE["minimum_margin"]) / 0.25
    )


def main():
    records = get_records()
    archives = [
        np.load(ROOT / f"experiments/v24_pseudo_validation/outputs/v23_warm60_pseudo_adapt_f{fold}.npz")
        for fold in (0, 1)
    ]
    prepared = [
        prepare_fold(
            archive["probabilities"],
            archive["indices"],
            archive["labels"],
            records,
            PAIR_SOURCES[fold],
            FOLDS[fold],
        )
        for fold, archive in enumerate(archives)
    ]
    candidates = []
    for template_strength, gap_penalty, maximum_edits, minimum_margin in product(
        (1.4, 1.8, 2.2),
        (0.6, 0.8, 1.0),
        (4, 5, 6),
        (0.0, 0.25),
    ):
        settings = {
            "template_strength": template_strength,
            "gap_penalty": gap_penalty,
            "maximum_edits": maximum_edits,
            "minimum_margin": minimum_margin,
            "allowed_users": PAIRED_USERS[0],
        }
        item = {key: value for key, value in settings.items() if key != "allowed_users"}
        item["fold0"] = score(prepared[0], settings)
        candidates.append(item)
        print(json.dumps(item), flush=True)
    selected = max(
        candidates,
        key=lambda item: (
            item["fold0"]["correct"],
            -item["fold0"]["changed_predictions"],
            -distance(item),
        ),
    )
    fold1_settings = {
        key: selected[key] for key in BASELINE
    } | {"allowed_users": PAIRED_USERS[1]}
    selected["fold1"] = score(prepared[1], fold1_settings)
    correct = selected["fold0"]["correct"] + selected["fold1"]["correct"]
    top = sorted(
        candidates,
        key=lambda item: (
            item["fold0"]["correct"],
            -item["fold0"]["changed_predictions"],
            -distance(item),
        ),
        reverse=True,
    )[:20]
    report = {
        "protocol": "Compact decoder grid selected on Fold 0; Fold 1 evaluated once blind.",
        "selected": selected,
        "combined": {"correct": correct, "rows": 1357, "accuracy": correct / 1357},
        "reference": {"correct": 1168, "rows": 1357},
        "promote": correct > 1168,
        "top_fold0": top,
    }
    output = ROOT / "experiments/v31_adapted_decoder_calibration/outputs/search.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "top_fold0"}, indent=2))


if __name__ == "__main__":
    main()

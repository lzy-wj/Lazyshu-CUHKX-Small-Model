#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.research_repeat_recovery_decoder import run
from src.data import get_records


SETTINGS = {
    "enabled": True,
    "maximum_gap": 150,
    "minimum_similarity": 0.7,
    "power": 1.0,
}


def load(path):
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def align(reference, candidate):
    positions = {int(index): position for position, index in enumerate(candidate["indices"])}
    order = np.asarray([positions[int(index)] for index in reference["indices"]])
    if not np.array_equal(reference["labels"], candidate["labels"][order]):
        raise ValueError("Validation labels do not align")
    return candidate["probabilities"][order].astype(np.float32)


def probabilities_for(reference, film, recipe):
    base = reference["probabilities"].astype(np.float32)
    if recipe["kind"] == "blend":
        weight = recipe["weight"]
        output = (1 - weight) * base + weight * film
    else:
        confidence = base.max(axis=1)
        disagreement = base.argmax(axis=1) != film.argmax(axis=1)
        selected = confidence < recipe["threshold"]
        if recipe["kind"] == "disagreement":
            selected &= disagreement
        output = base.copy()
        weight = recipe["weight"]
        output[selected] = (1 - weight) * base[selected] + weight * film[selected]
    return output / output.sum(axis=1, keepdims=True)


def main():
    records = get_records()
    folds = []
    for fold in (0, 1):
        reference = load(
            ROOT / f"experiments/v52_domain_logit_residual/outputs/gated_f{fold}.npz"
        )
        film = load(
            ROOT / f"experiments/v71_gated_film_distillation/outputs/student_f{fold}.npz"
        )
        folds.append((reference, align(reference, film)))
    recipes = [
        {"kind": "blend", "weight": weight}
        for weight in (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)
    ]
    for kind in ("confidence", "disagreement"):
        for threshold in (0.45, 0.55, 0.65, 0.75, 0.85):
            for weight in (0.35, 0.5, 0.75, 1.0):
                recipes.append(
                    {"kind": kind, "threshold": threshold, "weight": weight}
                )
    candidates = []
    for recipe in recipes:
        reference, film = folds[0]
        probabilities = probabilities_for(reference, film, recipe)
        result = run(
            0,
            (probabilities, reference["indices"], reference["labels"]),
            records,
            SETTINGS,
        )
        raw_correct = int(
            (probabilities.argmax(axis=1) == reference["labels"]).sum()
        )
        candidates.append(recipe | {"fold0": result, "fold0_raw_correct": raw_correct})
    selected = max(
        candidates,
        key=lambda item: (
            item["fold0"]["correct"],
            item["fold0_raw_correct"],
            -item["fold0"]["changed_predictions"],
            -item["weight"],
            item["kind"] == "blend",
        ),
    )
    recipe = {key: selected[key] for key in ("kind", "threshold", "weight") if key in selected}
    reference, film = folds[1]
    probabilities = probabilities_for(reference, film, recipe)
    selected["fold1"] = run(
        1,
        (probabilities, reference["indices"], reference["labels"]),
        records,
        SETTINGS,
    )
    selected["fold1_raw_correct"] = int(
        (probabilities.argmax(axis=1) == reference["labels"]).sum()
    )
    correct = selected["fold0"]["correct"] + selected["fold1"]["correct"]
    report = {
        "protocol": (
            "A compact V52-to-V71 confidence gate is selected only on Fold 0. "
            "Fold 1 uses the frozen gate and V59 decoder without inspection."
        ),
        "settings": SETTINGS,
        "selected": selected,
        "combined": {"correct": correct, "rows": 1357, "accuracy": correct / 1357},
        "reference_v71": {"correct": 1201, "rows": 1357},
        "promote": (
            selected["fold0"]["correct"] >= 570
            and selected["fold1"]["correct"] >= 631
            and correct > 1201
        ),
        "candidates": candidates,
    }
    output = HERE / "outputs/validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "candidates"}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.blend_aware_v12.audit_existing import temperature_scale
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


def metrics(probabilities, labels):
    predictions = probabilities.argmax(axis=1)
    return {
        "correct": int((predictions == labels).sum()),
        "rows": len(labels),
        "accuracy": float((predictions == labels).mean()),
        "nll": float(
            -np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1)).mean()
        ),
    }


def main():
    records = get_records()
    reference = load(ROOT / "experiments/v52_domain_logit_residual/outputs/gated_f1.npz")
    soup = load(HERE / "outputs/predictions/soup_w050_f1.npz")
    candidate = align(reference, soup)
    reports = []
    for weight in (0.0, 0.05, 0.1, 0.2, 0.3):
        blended = (
            (1 - weight) * reference["probabilities"]
            + weight * temperature_scale(candidate, 2.0)
        )
        reports.append(
            {
                "weight": weight,
                "raw": metrics(blended, reference["labels"]),
                "pair": run(
                    1,
                    (blended, reference["indices"], reference["labels"]),
                    records,
                    SETTINGS,
                ),
            }
        )
    selected = max(
        reports,
        key=lambda item: (
            item["pair"]["correct"],
            item["raw"]["correct"],
            -item["raw"]["nll"],
            -item["weight"],
        ),
    )
    fold0 = json.loads((HERE / "outputs/fold0_selection.json").read_text())["selected"]
    combined = fold0["selected_blend"]["pair"]["correct"] + selected["pair"]["correct"]
    report = {
        "protocol": "Fold-0-selected 50% V55/V65 parameter soup; Fold 1 is blind.",
        "fold0": fold0["selected_blend"],
        "fold1": selected,
        "combined": {"correct": combined, "rows": 1357, "accuracy": combined / 1357},
        "reference_v59": {"correct": 1197, "rows": 1357},
        "promote": combined > 1197 and selected["pair"]["correct"] >= 631,
        "candidates": reports,
    }
    output = HERE / "outputs/validation.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "candidates"}, indent=2))


if __name__ == "__main__":
    main()

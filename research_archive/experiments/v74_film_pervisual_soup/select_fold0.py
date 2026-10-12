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
    reference = load(ROOT / "experiments/v52_domain_logit_residual/outputs/gated_f0.npz")
    candidates = []
    for soup_weight in (0.25, 0.5, 0.75):
        tag = f"{round(soup_weight * 100):03d}"
        soup = load(HERE / f"outputs/predictions/soup_w{tag}_f0.npz")
        probabilities = align(reference, soup)
        soup_result = {
            "soup_v65_weight": soup_weight,
            "standalone": metrics(probabilities, reference["labels"]),
            "blends": [],
        }
        scaled = temperature_scale(probabilities, 2.0)
        for blend_weight in (0.0, 0.05, 0.1, 0.2, 0.3):
            blended = (
                (1 - blend_weight) * reference["probabilities"]
                + blend_weight * scaled
            )
            pair = run(
                0,
                (blended, reference["indices"], reference["labels"]),
                records,
                SETTINGS,
            )
            soup_result["blends"].append(
                {
                    "weight": blend_weight,
                    "raw": metrics(blended, reference["labels"]),
                    "pair": pair,
                }
            )
        soup_result["selected_blend"] = max(
            soup_result["blends"],
            key=lambda item: (
                item["pair"]["correct"],
                item["raw"]["correct"],
                -item["raw"]["nll"],
                -item["weight"],
            ),
        )
        candidates.append(soup_result)
    selected = max(
        candidates,
        key=lambda item: (
            item["selected_blend"]["pair"]["correct"],
            item["selected_blend"]["raw"]["correct"],
            item["standalone"]["correct"],
            -item["selected_blend"]["raw"]["nll"],
        ),
    )
    report = {
        "protocol": (
            "V55/V65 soup and V52 blend weights are selected on Fold 0 only. "
            "Temperature 2 and the V59 decoder remain frozen."
        ),
        "selected": selected,
        "reference_v52_pair_correct": 568,
        "advance_fold1": (
            selected["standalone"]["correct"] >= 475
            or selected["selected_blend"]["pair"]["correct"] > 568
        ),
        "candidates": candidates,
    }
    output = HERE / "outputs/fold0_selection.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "candidates"}, indent=2))


if __name__ == "__main__":
    main()

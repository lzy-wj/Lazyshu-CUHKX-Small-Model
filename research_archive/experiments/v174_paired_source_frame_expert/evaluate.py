#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.research_repeat_recovery_decoder import run
from experiments.v160_radar_batch_decoder.radar_batch import install_radar_batch_intervals
from experiments.v166_repeated_transition_consensus.evaluate import fold_inputs
from experiments.v168_modality_distribution_alignment.evaluate import v166
from experiments.v70_film_margin_distillation.validate import SETTINGS
from src.data import get_records


TARGETS = ((0, 8), (0, 18), (1, 7), (1, 17))
WEIGHTS = (0.1, 0.2, 0.3, 0.5)
TEMPERATURES = (0.5, 1.0, 1.5)


def metrics(prediction, labels):
    return {
        "correct": int((prediction == labels).sum()),
        "rows": len(labels),
        "accuracy": float((prediction == labels).mean()),
    }


def comparison(prediction, baseline, labels):
    changed = prediction != baseline
    return {
        "changed": int(changed.sum()),
        "gains": int((changed & (prediction == labels) & (baseline != labels)).sum()),
        "losses": int((changed & (prediction != labels) & (baseline == labels)).sum()),
        "delta": int((prediction == labels).sum() - (baseline == labels).sum()),
    }


def evaluate_target(fold, target_user, records):
    path = HERE / f"outputs/v174_thermal_user{target_user}_f{fold}/validation.npz"
    if not path.exists():
        return {"fold": fold, "target_user": target_user, "status": "training"}
    with np.load(path) as archive:
        thermal = {key: archive[key].copy() for key in archive.files}
    reference, evidence, _, _ = fold_inputs(fold, records)
    _, base = v166(fold, records)
    positions = {
        int(index): position for position, index in enumerate(reference["indices"])
    }
    target_positions = np.asarray(
        [positions[int(index)] for index in thermal["indices"]], dtype=np.int64
    )
    if any(records[int(index)]["user"] != target_user for index in thermal["indices"]):
        raise ValueError("Validation archive contains a non-target user")
    labels = reference["labels"].astype(np.int64)
    target_labels = labels[target_positions]
    raw = thermal["probabilities"].argmax(axis=1)
    base_target = base[target_positions]
    routes = []
    for temperature in TEMPERATURES:
        scaled = np.clip(thermal["probabilities"].astype(np.float64), 1e-12, 1.0)
        scaled = scaled ** (1 / temperature)
        scaled /= scaled.sum(axis=1, keepdims=True)
        expanded = evidence.copy()
        expanded[target_positions] = scaled
        for weight in WEIGHTS:
            probabilities = (1 - weight) * evidence + weight * expanded
            prediction = np.asarray(
                run(
                    fold,
                    (probabilities, reference["indices"], labels),
                    records,
                    SETTINGS,
                )["predictions"],
                dtype=np.int64,
            )[target_positions]
            routes.append(
                {
                    "temperature": temperature,
                    "thermal_weight": weight,
                    "metrics": metrics(prediction, target_labels),
                    "versus_v166": comparison(
                        prediction, base_target, target_labels
                    ),
                }
            )
    return {
        "fold": fold,
        "target_user": target_user,
        "status": "complete",
        "v166": metrics(base_target, target_labels),
        "thermal_raw": metrics(raw, target_labels),
        "oracle_correct": int(
            ((raw == target_labels) | (base_target == target_labels)).sum()
        ),
        "routes": routes,
    }


def main():
    install_radar_batch_intervals()
    records = get_records()
    targets = [
        evaluate_target(fold, target_user, records)
        for fold, target_user in TARGETS
    ]
    report = {
        "protocol": (
            "Each model excludes only its target alias while retaining the "
            "same-person opposite-environment source. Thermal evidence replaces "
            "only that target and uses the frozen paired decoder."
        ),
        "targets": targets,
    }
    output = HERE / "outputs/evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

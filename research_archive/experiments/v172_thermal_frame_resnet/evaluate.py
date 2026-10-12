#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.research_repeat_recovery_decoder import run
from experiments.v125_domain_complement.evaluate import scores
from experiments.v160_radar_batch_decoder.radar_batch import install_radar_batch_intervals
from experiments.v166_repeated_transition_consensus.evaluate import fold_inputs
from experiments.v168_modality_distribution_alignment.evaluate import route_report, v166
from experiments.v70_film_margin_distillation.validate import SETTINGS
from src.data import get_records


WEIGHTS = (0.1, 0.2, 0.3, 0.4, 0.5)
TEMPERATURES = (0.5, 1.0, 1.5)


def load_archive(path):
    with np.load(path) as archive:
        return {key: archive[key].copy() for key in archive.files}


def evaluate_fold(fold, records):
    path = HERE / f"outputs/v172_thermal_resnet34_f{fold}/validation.npz"
    if not path.exists():
        return {"fold": fold, "status": "training"}
    thermal = load_archive(path)
    reference, evidence, _, _ = fold_inputs(fold, records)
    frozen_reference, base = v166(fold, records)
    if not np.array_equal(reference["indices"], frozen_reference["indices"]):
        raise ValueError("Fold references do not align")
    positions = {
        int(index): position for position, index in enumerate(reference["indices"])
    }
    thermal_positions = np.asarray(
        [positions[int(index)] for index in thermal["indices"]], dtype=np.int64
    )
    if not np.array_equal(
        reference["labels"][thermal_positions].astype(np.int64),
        thermal["labels"].astype(np.int64),
    ):
        raise ValueError("Thermal labels do not align with the fold reference")
    labels = reference["labels"].astype(np.int64)
    raw = thermal["probabilities"].argmax(axis=1)
    base_subset = base[thermal_positions]
    raw_changed = raw != base_subset
    raw_report = {
        "thermal": {
            "correct": int((raw == thermal["labels"]).sum()),
            "rows": len(raw),
            "accuracy": float((raw == thermal["labels"]).mean()),
        },
        "v166": {
            "correct": int((base_subset == thermal["labels"]).sum()),
            "rows": len(raw),
            "accuracy": float((base_subset == thermal["labels"]).mean()),
        },
        "oracle_correct": int(
            ((raw == thermal["labels"]) | (base_subset == thermal["labels"])).sum()
        ),
        "thermal_only_gains": int(
            ((raw == thermal["labels"]) & (base_subset != thermal["labels"])).sum()
        ),
        "changed": int(raw_changed.sum()),
    }
    routes = []
    for temperature in TEMPERATURES:
        scaled = np.clip(thermal["probabilities"].astype(np.float64), 1e-12, 1.0)
        scaled = scaled ** (1.0 / temperature)
        scaled /= scaled.sum(axis=1, keepdims=True)
        expanded = evidence.copy()
        expanded[thermal_positions] = scaled
        for weight in WEIGHTS:
            probabilities = (1 - weight) * evidence + weight * expanded
            decoded = np.asarray(
                run(
                    fold,
                    (probabilities, reference["indices"], labels),
                    records,
                    SETTINGS,
                )["predictions"],
                dtype=np.int64,
            )
            routes.append(
                {
                    "temperature": temperature,
                    "thermal_weight": weight,
                    "all": route_report(
                        decoded, base, reference, fold, records, "all"
                    ),
                    "unmatched": route_report(
                        decoded, base, reference, fold, records, "unmatched"
                    ),
                }
            )
    return {
        "fold": fold,
        "status": "complete",
        "v166": scores(base, reference, fold, records),
        "raw": raw_report,
        "routes": routes,
    }


def main():
    install_radar_batch_intervals()
    records = get_records()
    folds = [evaluate_fold(fold, records) for fold in (0, 1)]
    report = {
        "protocol": (
            "Thermal clip probabilities replace the frozen system evidence only "
            "where Thermal frames exist. A fixed coarse temperature/weight grid "
            "is Radar-batch decoded and compared against V166."
        ),
        "folds": folds,
    }
    output = HERE / "outputs/evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

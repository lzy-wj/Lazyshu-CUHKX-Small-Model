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
from experiments.v168_modality_distribution_alignment.evaluate import (
    aligned_probabilities,
    route_report,
    v166,
)
from experiments.v70_film_margin_distillation.validate import SETTINGS
from src.data import get_records


WEIGHTS = (0.25, 0.5, 0.75, 1.0)


def evaluate_fold(fold, records):
    path = HERE / f"outputs/checkpoints/v171_full6_160_f{fold}/validation.npz"
    if not path.exists():
        return {"fold": fold, "status": "training"}
    reference, base = v166(fold, records)
    candidate = aligned_probabilities(path, reference["indices"])
    v71 = aligned_probabilities(
        ROOT / f"experiments/v71_gated_film_distillation/outputs/student_f{fold}.npz",
        reference["indices"],
    )
    labels = reference["labels"].astype(np.int64)
    routes = []
    for weight in WEIGHTS:
        probabilities = (1 - weight) * v71 + weight * candidate
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
                "candidate_weight": weight,
                "unmatched": route_report(
                    decoded, base, reference, fold, records, "unmatched"
                ),
                "all": route_report(decoded, base, reference, fold, records, "all"),
            }
        )
    raw = candidate.argmax(axis=1)
    return {
        "fold": fold,
        "status": "complete",
        "v166": scores(base, reference, fold, records),
        "raw": scores(raw, reference, fold, records),
        "routes": routes,
    }


def main():
    install_radar_batch_intervals()
    records = get_records()
    folds = [evaluate_fold(fold, records) for fold in (0, 1)]
    report = {
        "protocol": (
            "V168 full-modality training at 160px; frozen V71 blend grid and "
            "Radar-batch decoder; every route is compared against V166."
        ),
        "folds": folds,
    }
    output = HERE / "outputs/evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

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
from experiments.v168_modality_distribution_alignment.evaluate import v166
from experiments.v70_film_margin_distillation.validate import SETTINGS
from src.data import get_records


WEIGHTS = (0.01, 0.02, 0.05, 0.1, 0.2)
TEMPERATURES = (0.5, 1.0, 2.0)


def aligned_probabilities(path, indices):
    with np.load(path) as archive:
        positions = {
            int(index): position for position, index in enumerate(archive["indices"])
        }
        return archive["probabilities"][
            np.asarray([positions[int(index)] for index in indices])
        ].astype(np.float64)


def scale(probabilities, temperature):
    probabilities = np.clip(probabilities, 1e-12, 1)
    probabilities = probabilities ** (1 / temperature)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def comparison(prediction, baseline, labels):
    changed = prediction != baseline
    return {
        "correct": int((prediction == labels).sum()),
        "changed": int(changed.sum()),
        "gains": int((changed & (prediction == labels) & (baseline != labels)).sum()),
        "losses": int((changed & (prediction != labels) & (baseline == labels)).sum()),
        "delta": int((prediction == labels).sum() - (baseline == labels).sum()),
    }


def evaluate_fold(fold, records):
    reference, evidence, _, _ = fold_inputs(fold, records)
    _, baseline = v166(fold, records)
    labels = reference["labels"].astype(np.int64)
    candidate = aligned_probabilities(
        HERE / f"outputs/v176_fullscene_f{fold}/validation.npz",
        reference["indices"],
    )
    raw = candidate.argmax(axis=1)
    decoded = np.asarray(
        run(
            fold,
            (candidate, reference["indices"], labels),
            records,
            SETTINGS,
        )["predictions"],
        dtype=np.int64,
    )
    routes = []
    for temperature in TEMPERATURES:
        adjusted = scale(candidate, temperature)
        for weight in WEIGHTS:
            probabilities = (1 - weight) * evidence + weight * adjusted
            prediction = np.asarray(
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
                    "weight": weight,
                    **comparison(prediction, baseline, labels),
                }
            )
    return {
        "fold": fold,
        "v166": scores(baseline, reference, fold, records),
        "raw": comparison(raw, baseline, labels),
        "decoded": comparison(decoded, baseline, labels),
        "raw_oracle_correct": int(
            ((raw == labels) | (baseline == labels)).sum()
        ),
        "decoded_oracle_correct": int(
            ((decoded == labels) | (baseline == labels)).sum()
        ),
        "routes": routes,
    }


def main():
    install_radar_batch_intervals()
    records = get_records()
    folds = [evaluate_fold(fold, records) for fold in (0, 1)]
    report = {
        "protocol": (
            "Full-scene probabilities are aligned to frozen V166 evidence, "
            "blended at small fixed weights, and passed through the unchanged "
            "Radar-batch sequence decoder."
        ),
        "folds": folds,
    }
    output = HERE / "outputs/evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

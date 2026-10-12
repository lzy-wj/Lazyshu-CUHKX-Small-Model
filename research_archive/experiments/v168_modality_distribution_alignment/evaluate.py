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
from experiments.v160_radar_batch_decoder.radar_batch import (
    install_radar_batch_intervals,
)
from experiments.v164_evidence_gated_radar.evaluate import accept
from experiments.v165_balanced_cycle_radar.evaluate import cycle_mask
from experiments.v166_repeated_transition_consensus.evaluate import (
    fold_inputs,
    repeated_transition_mask,
)
from experiments.v70_film_margin_distillation.validate import SETTINGS
from experiments.v84_residual_invariant_skeleton.evaluate import domains
from src.data import get_records


VARIANTS = (
    "minimum5",
    "full6",
    "minimum5_retrain",
    "full6_retrain",
)
WEIGHTS = (0.25, 0.5, 0.75, 1.0)


def aligned_probabilities(path, indices):
    with np.load(path) as archive:
        positions = {
            int(index): position
            for position, index in enumerate(archive["indices"])
        }
        return archive["probabilities"][
            np.asarray([positions[int(index)] for index in indices])
        ].astype(np.float64)


def v166(fold, records):
    reference, evidence, incumbent, challenger = fold_inputs(fold, records)
    users = [records[int(index)]["user"] for index in reference["indices"]]
    selected = cycle_mask(
        reference["indices"], incumbent, challenger, records
    )
    selected |= accept(incumbent, challenger, evidence)
    selected |= repeated_transition_mask(users, incumbent, challenger, 2)
    prediction = incumbent.copy()
    prediction[selected] = challenger[selected]
    return reference, prediction


def subset_accuracy(prediction, reference, records, minimum_modalities):
    selected = np.asarray(
        [
            len(records[int(index)]["modalities"]) >= minimum_modalities
            for index in reference["indices"]
        ],
        dtype=bool,
    )
    return {
        "correct": int((prediction[selected] == reference["labels"][selected]).sum()),
        "rows": int(selected.sum()),
        "accuracy": float(
            (prediction[selected] == reference["labels"][selected]).mean()
        ),
    }


def route_report(hard, base, reference, fold, records, route):
    domain = domains(reference["indices"], fold, records)
    selected = domain == 1 if route == "unmatched" else np.ones(len(domain), dtype=bool)
    prediction = base.copy()
    prediction[selected] = hard[selected]
    changed = selected & (prediction != base)
    labels = reference["labels"].astype(np.int64)
    return {
        "route": route,
        "changed": int(changed.sum()),
        "gains": int((changed & (prediction == labels) & (base != labels)).sum()),
        "losses": int((changed & (prediction != labels) & (base == labels)).sum()),
        "metrics": scores(prediction, reference, fold, records),
        "minimum5": subset_accuracy(prediction, reference, records, 5),
        "full6": subset_accuracy(prediction, reference, records, 6),
    }


def evaluate_variant(name, fold, reference, base, records, v71):
    path = HERE / f"outputs/checkpoints/v168_{name}_f{fold}/validation.npz"
    if not path.exists():
        return None
    candidate = aligned_probabilities(path, reference["indices"])
    labels = reference["labels"].astype(np.int64)
    variants = []
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
        variants.append(
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
        "archive": str(path.relative_to(ROOT)),
        "raw": {
            "correct": int((raw == labels).sum()),
            "rows": len(labels),
            "accuracy": float((raw == labels).mean()),
            "minimum5": subset_accuracy(raw, reference, records, 5),
            "full6": subset_accuracy(raw, reference, records, 6),
        },
        "routes": variants,
    }


def main():
    install_radar_batch_intervals()
    records = get_records()
    folds = []
    for fold in (0, 1):
        reference, base = v166(fold, records)
        v71 = aligned_probabilities(
            ROOT
            / "experiments/v71_gated_film_distillation/outputs"
            / f"student_f{fold}.npz",
            reference["indices"],
        )
        variants = {
            name: evaluate_variant(
                name, fold, reference, base, records, v71
            )
            for name in VARIANTS
        }
        variants = {name: value for name, value in variants.items() if value}
        folds.append(
            {
                "fold": fold,
                "v166": scores(base, reference, fold, records),
                "variants": variants,
            }
        )
    report = {
        "protocol": (
            "Availability-aligned checkpoints are blended with frozen V71, "
            "decoded with frozen Radar intervals, and compared against V166."
        ),
        "folds": folds,
    }
    output = HERE / "outputs/evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

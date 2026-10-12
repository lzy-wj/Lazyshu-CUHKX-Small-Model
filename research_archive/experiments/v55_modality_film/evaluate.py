#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.blend_aware_v12.audit_existing import temperature_scale
from experiments.scene_context_v13.research_repeat_recovery_decoder import run
from src.data import get_records


HERE = Path(__file__).resolve().parent
WEIGHTS = (0.0, 0.05, 0.1, 0.2, 0.3)
PAIR_SETTINGS = {
    "enabled": False,
    "maximum_gap": 120,
    "minimum_similarity": 0.7,
    "power": 1.5,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    return parser.parse_args()


def load(path):
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def align(reference, candidate):
    positions = {int(index): position for position, index in enumerate(candidate["indices"])}
    order = np.asarray([positions[int(index)] for index in reference["indices"]])
    if not np.array_equal(reference["labels"], candidate["labels"][order]):
        raise ValueError("Validation labels do not align")
    return candidate["probabilities"][order].astype(np.float64)


def raw_metrics(probabilities, labels):
    predictions = probabilities.argmax(axis=1)
    return {
        "correct": int((predictions == labels).sum()),
        "rows": len(labels),
        "accuracy": float((predictions == labels).mean()),
        "nll": float(-np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1)).mean()),
    }


def pair_metrics(fold, probabilities, indices, labels, records):
    result = run(fold, (probabilities, indices, labels), records, PAIR_SETTINGS)
    return {key: result[key] for key in ("correct", "accuracy", "changed_predictions", "rounds")}


def main():
    args = parse_args()
    fold = args.fold
    records = get_records()
    reference = load(
        ROOT
        / f"experiments/scene_context_v13/outputs/checkpoints/v18_shared5_joint_noradar_f{fold}/validation.npz"
    )
    candidate = load(HERE / f"outputs/checkpoints/v55_modality_film_f{fold}/validation.npz")
    v52 = load(ROOT / f"experiments/v52_domain_logit_residual/outputs/gated_f{fold}.npz")
    candidate_probabilities = align(reference, candidate)
    v52_probabilities = align(reference, v52)
    labels = reference["labels"].astype(np.int64)
    indices = reference["indices"].astype(np.int64)
    reference_probabilities = reference["probabilities"].astype(np.float64)
    reference_predictions = reference_probabilities.argmax(axis=1)
    candidate_predictions = candidate_probabilities.argmax(axis=1)
    corrections = {
        "candidate_only_correct": int(
            ((candidate_predictions == labels) & (reference_predictions != labels)).sum()
        ),
        "reference_only_correct": int(
            ((reference_predictions == labels) & (candidate_predictions != labels)).sum()
        ),
        "changed_predictions": int((candidate_predictions != reference_predictions).sum()),
    }
    blend_candidates = []
    scaled_candidate = temperature_scale(candidate_probabilities, 2.0)
    weights = WEIGHTS if fold == 0 else (
        json.loads((HERE / "outputs/fold0_evaluation.json").read_text())["selected_blend"]["weight"],
    )
    for weight in weights:
        probabilities = (1 - weight) * v52_probabilities + weight * scaled_candidate
        blend_candidates.append(
            {
                "weight": weight,
                "raw": raw_metrics(probabilities, labels),
                "pair": pair_metrics(fold, probabilities, indices, labels, records),
            }
        )
    selected_blend = max(
        blend_candidates,
        key=lambda item: (item["pair"]["correct"], item["raw"]["correct"], -item["weight"]),
    )
    report = {
        "protocol": (
            "V55 changes only the joint visual encoder by adding zero-initialized modality-specific "
            "channel affine adapters. Candidate temperature is fixed at 2.0; Fold 0 selects a "
            "coarse V52 blend weight and Fold 1 reuses it unchanged."
        ),
        "fold": fold,
        "reference_v18": {
            "raw": raw_metrics(reference_probabilities, labels),
            "pair": pair_metrics(fold, reference_probabilities, indices, labels, records),
        },
        "candidate_v55": {
            "raw": raw_metrics(candidate_probabilities, labels),
            "pair": pair_metrics(fold, candidate_probabilities, indices, labels, records),
            "corrections_vs_v18": corrections,
        },
        "reference_v52": {
            "raw": raw_metrics(v52_probabilities, labels),
            "pair": pair_metrics(fold, v52_probabilities, indices, labels, records),
        },
        "selected_blend": selected_blend,
        "blend_candidates": blend_candidates,
    }
    if fold == 0:
        improved_plateau = sum(
            item["pair"]["correct"] > report["reference_v52"]["pair"]["correct"]
            for item in blend_candidates
        ) >= 2
        report["advance_fold1"] = (
            report["candidate_v55"]["raw"]["correct"]
            > report["reference_v18"]["raw"]["correct"]
            or selected_blend["pair"]["correct"] >= report["reference_v52"]["pair"]["correct"] + 2
            or (
                selected_blend["pair"]["correct"] > report["reference_v52"]["pair"]["correct"]
                and improved_plateau
                and selected_blend["raw"]["nll"]
                < 0.9 * report["reference_v52"]["raw"]["nll"]
            )
        )
        report["fold1_gate_evidence"] = {
            "improved_blend_weights": sum(
                item["pair"]["correct"] > report["reference_v52"]["pair"]["correct"]
                for item in blend_candidates
            ),
            "selected_nll_ratio_vs_v52": (
                selected_blend["raw"]["nll"] / report["reference_v52"]["raw"]["nll"]
            ),
        }
    else:
        fold0 = json.loads((HERE / "outputs/fold0_evaluation.json").read_text())
        report["combined"] = {
            "v18_raw_correct": (
                fold0["reference_v18"]["raw"]["correct"]
                + report["reference_v18"]["raw"]["correct"]
            ),
            "v55_raw_correct": (
                fold0["candidate_v55"]["raw"]["correct"]
                + report["candidate_v55"]["raw"]["correct"]
            ),
            "v52_pair_correct": (
                fold0["reference_v52"]["pair"]["correct"]
                + report["reference_v52"]["pair"]["correct"]
            ),
            "v55_blend_pair_correct": (
                fold0["selected_blend"]["pair"]["correct"]
                + report["selected_blend"]["pair"]["correct"]
            ),
            "rows": 1357,
        }
        report["promote"] = (
            report["combined"]["v55_blend_pair_correct"]
            > report["combined"]["v52_pair_correct"]
            and report["selected_blend"]["pair"]["correct"]
            >= report["reference_v52"]["pair"]["correct"]
        )
    output = HERE / f"outputs/fold{fold}_evaluation.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "blend_candidates"}, indent=2))


if __name__ == "__main__":
    main()

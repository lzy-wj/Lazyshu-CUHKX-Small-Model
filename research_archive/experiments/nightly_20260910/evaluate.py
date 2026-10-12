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


CANDIDATES = {
    "v61_film_seed43": "v61_film_seed43_f{fold}",
    "v62_film_highres160": "v62_film_highres160_f{fold}",
    "v63_film_maskdistill": "v63_film_maskdistill_f{fold}",
    "v64_film_invariant_skeleton": "v64_film_invariant_skeleton_f{fold}",
    "v65_film_pervisual_aux": "v65_film_pervisual_aux_f{fold}",
}
WEIGHTS = (0.0, 0.05, 0.1, 0.2, 0.3)
PAIR_SETTINGS = {
    "enabled": False,
    "maximum_gap": 120,
    "minimum_similarity": 0.7,
    "power": 1.5,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slug", choices=tuple(CANDIDATES), required=True)
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


def metrics(probabilities, labels, indices, records):
    predictions = probabilities.argmax(axis=1)
    users = np.asarray([records[int(index)]["user"] for index in indices])
    user_accuracies = [
        float((predictions[users == user] == labels[users == user]).mean())
        for user in np.unique(users)
    ]
    return {
        "correct": int((predictions == labels).sum()),
        "rows": len(labels),
        "accuracy": float((predictions == labels).mean()),
        "nll": float(
            -np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1)).mean()
        ),
        "user_macro_accuracy": float(np.mean(user_accuracies)),
        "user_min_accuracy": float(np.min(user_accuracies)),
    }


def pair_metrics(fold, probabilities, indices, labels, records):
    result = run(fold, (probabilities, indices, labels), records, PAIR_SETTINGS)
    return {key: result[key] for key in ("correct", "accuracy", "changed_predictions", "rounds")}


def main():
    args = parse_args()
    fold = args.fold
    records = get_records()
    experiment = ROOT / "experiments" / args.slug
    name = CANDIDATES[args.slug].format(fold=fold)
    reference = load(
        ROOT
        / f"experiments/scene_context_v13/outputs/checkpoints/"
        f"v18_shared5_joint_noradar_f{fold}/validation.npz"
    )
    candidate = load(experiment / f"outputs/checkpoints/{name}/validation.npz")
    v52 = load(ROOT / f"experiments/v52_domain_logit_residual/outputs/gated_f{fold}.npz")
    labels = reference["labels"].astype(np.int64)
    indices = reference["indices"].astype(np.int64)
    reference_probabilities = reference["probabilities"].astype(np.float64)
    candidate_probabilities = align(reference, candidate)
    v52_probabilities = align(reference, v52)
    reference_predictions = reference_probabilities.argmax(axis=1)
    candidate_predictions = candidate_probabilities.argmax(axis=1)
    scaled_candidate = temperature_scale(candidate_probabilities, 2.0)
    if fold == 0:
        weights = WEIGHTS
    else:
        fold0 = json.loads((experiment / "outputs/fold0_evaluation.json").read_text())
        weights = (fold0["selected_blend"]["weight"],)
    blend_candidates = []
    for weight in weights:
        probabilities = (1 - weight) * v52_probabilities + weight * scaled_candidate
        blend_candidates.append(
            {
                "weight": weight,
                "raw": metrics(probabilities, labels, indices, records),
                "pair": pair_metrics(fold, probabilities, indices, labels, records),
            }
        )
    selected_blend = max(
        blend_candidates,
        key=lambda item: (item["pair"]["correct"], item["raw"]["correct"], -item["weight"]),
    )
    report = {
        "protocol": (
            "Fixed temperature 2.0 and V52 blend grid. Fold 0 selects one blend weight; "
            "Fold 1 reuses it unchanged. The pair decoder and all references are frozen."
        ),
        "slug": args.slug,
        "name": name,
        "fold": fold,
        "reference_v18": {
            "raw": metrics(reference_probabilities, labels, indices, records),
            "pair": pair_metrics(fold, reference_probabilities, indices, labels, records),
        },
        "candidate": {
            "raw": metrics(candidate_probabilities, labels, indices, records),
            "pair": pair_metrics(fold, candidate_probabilities, indices, labels, records),
            "corrections_vs_v18": {
                "candidate_only_correct": int(
                    ((candidate_predictions == labels) & (reference_predictions != labels)).sum()
                ),
                "reference_only_correct": int(
                    ((reference_predictions == labels) & (candidate_predictions != labels)).sum()
                ),
                "changed_predictions": int((candidate_predictions != reference_predictions).sum()),
            },
        },
        "reference_v52": {
            "raw": metrics(v52_probabilities, labels, indices, records),
            "pair": pair_metrics(fold, v52_probabilities, indices, labels, records),
        },
        "selected_blend": selected_blend,
        "blend_candidates": blend_candidates,
    }
    if fold == 0:
        improved_weights = sum(
            item["pair"]["correct"] > report["reference_v52"]["pair"]["correct"]
            for item in blend_candidates
        )
        nll_ratio = selected_blend["raw"]["nll"] / report["reference_v52"]["raw"]["nll"]
        report["advance_fold1"] = (
            report["candidate"]["raw"]["correct"]
            > report["reference_v18"]["raw"]["correct"]
            or selected_blend["pair"]["correct"]
            >= report["reference_v52"]["pair"]["correct"] + 2
            or (
                selected_blend["pair"]["correct"]
                > report["reference_v52"]["pair"]["correct"]
                and improved_weights >= 2
                and nll_ratio < 0.9
            )
        )
        report["fold1_gate_evidence"] = {
            "improved_blend_weights": improved_weights,
            "selected_nll_ratio_vs_v52": nll_ratio,
        }
    else:
        fold0 = json.loads((experiment / "outputs/fold0_evaluation.json").read_text())
        report["combined"] = {
            "candidate_raw_correct": (
                fold0["candidate"]["raw"]["correct"] + report["candidate"]["raw"]["correct"]
            ),
            "v52_pair_correct": (
                fold0["reference_v52"]["pair"]["correct"]
                + report["reference_v52"]["pair"]["correct"]
            ),
            "candidate_blend_pair_correct": (
                fold0["selected_blend"]["pair"]["correct"]
                + report["selected_blend"]["pair"]["correct"]
            ),
            "rows": 1357,
        }
        report["promote"] = (
            report["combined"]["candidate_blend_pair_correct"]
            > report["combined"]["v52_pair_correct"]
            and report["selected_blend"]["pair"]["correct"]
            >= report["reference_v52"]["pair"]["correct"]
        )
    output = experiment / f"outputs/fold{fold}_evaluation.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "blend_candidates"}, indent=2))


if __name__ == "__main__":
    main()


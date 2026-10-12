#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.research_repeat_recovery_decoder import prepool
from experiments.scene_context_v13.research_sequence_decoder import repeat_pool
from experiments.v160_radar_batch_decoder.radar_batch import (
    install_radar_batch_intervals,
)
from experiments.v168_modality_distribution_alignment.evaluate import v166
from experiments.v70_film_margin_distillation.validate import SETTINGS
from scripts.evaluate_repeat_decoder import assign_transition_unique, transition_matrix
from src.data import get_records


TARGETS = ((0, 23), (1, 22))
WEIGHTS = (0.25, 0.5, 0.75, 1.0)


def load_archive(path):
    with np.load(path) as archive:
        return {key: archive[key].copy() for key in archive.files}


def aligned(values, indices, key="probabilities"):
    positions = {
        int(index): position for position, index in enumerate(values["indices"])
    }
    return values[key][
        np.asarray([positions[int(index)] for index in indices])
    ].copy()


def decode(probabilities, indices, records, target_user):
    probabilities, recovered_groups = prepool(
        probabilities,
        indices,
        records,
        SETTINGS["maximum_gap"],
        SETTINGS["minimum_similarity"],
        SETTINGS["power"],
    )
    probability_by_index, round_values, repeat_groups = repeat_pool(
        probabilities, indices, records
    )
    transition = transition_matrix(
        records,
        {target_user},
        environment=int(target_user > 15),
    )
    prediction_by_index = {}
    for item in round_values:
        values = np.stack(
            [probability_by_index[int(index)] for index in item["rows"]]
        )
        assignment = assign_transition_unique(
            values, transition, strength=0.4, beam_size=100
        ).argmax(axis=1)
        prediction_by_index.update(
            (int(index), int(label))
            for index, label in zip(item["rows"], assignment)
        )
    predictions = np.asarray(
        [
            prediction_by_index.get(int(index), int(probability.argmax()))
            for index, probability in zip(indices, probabilities)
        ],
        dtype=np.int64,
    )
    return predictions, {
        "rounds": len(round_values),
        "recovered_groups": recovered_groups,
        "repeat_groups": len(repeat_groups),
    }


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
    candidate_path = (
        HERE
        / "outputs/checkpoints"
        / f"v169_user{target_user}_f{fold}"
        / "validation.npz"
    )
    if not candidate_path.exists():
        return {
            "fold": fold,
            "target_user": target_user,
            "status": "training",
            "archive": str(candidate_path.relative_to(ROOT)),
        }

    candidate = load_archive(candidate_path)
    indices = candidate["indices"].astype(np.int64)
    labels = candidate["labels"].astype(np.int64)
    expected = np.asarray(
        [
            record["index"]
            for record in records
            if record["split"] == "train" and record["user"] == target_user
        ],
        dtype=np.int64,
    )
    if set(indices.tolist()) != set(expected.tolist()):
        raise ValueError(f"Target user {target_user} validation rows do not align")
    if any(records[int(index)]["user"] != target_user for index in indices):
        raise ValueError(f"Validation archive leaks users other than {target_user}")

    candidate_probabilities = candidate["probabilities"].astype(np.float64)
    v71_archive = load_archive(
        ROOT
        / "experiments/v71_gated_film_distillation/outputs"
        / f"student_f{fold}.npz"
    )
    v23_archive = load_archive(
        ROOT
        / "experiments/scene_context_v13/outputs/checkpoints"
        / f"v23_shared5_joint_warm_maskdistill_f{fold}"
        / "validation.npz"
    )
    v71_probabilities = aligned(v71_archive, indices).astype(np.float64)
    v23_probabilities = aligned(v23_archive, indices).astype(np.float64)

    full_reference, baseline_predictions = v166(fold, records)
    baseline_values = {
        "indices": full_reference["indices"],
        "pseudo_labels": baseline_predictions,
    }
    v166_predictions = aligned(
        baseline_values, indices, key="pseudo_labels"
    ).astype(np.int64)

    sources = {
        "v71": v71_probabilities,
        "v23": v23_probabilities,
        "candidate": candidate_probabilities,
    }
    source_reports = {}
    for name, probabilities in sources.items():
        raw = probabilities.argmax(axis=1).astype(np.int64)
        decoded, decoder_report = decode(
            probabilities, indices, records, target_user
        )
        source_reports[name] = {
            "raw": metrics(raw, labels),
            "decoded": metrics(decoded, labels),
            "versus_v166": comparison(decoded, v166_predictions, labels),
            "decoder": decoder_report,
        }

    blends = []
    for anchor_name, anchor in (("v71", v71_probabilities), ("v23", v23_probabilities)):
        for weight in WEIGHTS:
            probabilities = (1.0 - weight) * anchor + weight * candidate_probabilities
            prediction, decoder_report = decode(
                probabilities, indices, records, target_user
            )
            blends.append(
                {
                    "anchor": anchor_name,
                    "candidate_weight": weight,
                    "metrics": metrics(prediction, labels),
                    "versus_v166": comparison(
                        prediction, v166_predictions, labels
                    ),
                    "decoder": decoder_report,
                }
            )

    return {
        "fold": fold,
        "target_user": target_user,
        "status": "complete",
        "archive": str(candidate_path.relative_to(ROOT)),
        "v166": metrics(v166_predictions, labels),
        "sources": source_reports,
        "blends": blends,
    }


def main():
    install_radar_batch_intervals()
    records = get_records()
    targets = [
        evaluate_target(fold, target_user, records)
        for fold, target_user in TARGETS
    ]
    complete = [item for item in targets if item["status"] == "complete"]
    report = {
        "protocol": (
            "Each target user is the only validation user. Optimization and the "
            "environment-specific transition prior exclude that target user. "
            "Candidate weights are a fixed grid and are reported against frozen V166."
        ),
        "targets": targets,
        "summary": {
            "complete_targets": len(complete),
            "v166_correct": sum(item["v166"]["correct"] for item in complete),
            "rows": sum(item["v166"]["rows"] for item in complete),
        },
    }
    output = HERE / "outputs/evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

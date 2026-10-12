#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.research_repeat_recovery_decoder import run
from src.data import get_records


DOMAIN_WEIGHTS = {0: 0.0, 1: 76 / 405, 2: 199 / 405, 3: 130 / 405}
PAIR_SETTINGS = {
    "enabled": False,
    "maximum_gap": 120,
    "minimum_similarity": 0.7,
    "power": 1.5,
}
PAIRED_USERS = ({8, 18}, {7, 17})


def load(path):
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def align(reference, candidate):
    positions = {int(index): position for position, index in enumerate(candidate["indices"])}
    order = np.asarray([positions[int(index)] for index in reference["indices"]])
    if not np.array_equal(reference["labels"], candidate["labels"][order]):
        raise ValueError("Validation labels do not align")
    return candidate["probabilities"][order].astype(np.float64)


def domains(indices, fold, records):
    return np.asarray(
        [
            int(records[int(index)]["user"] > 15)
            + 2 * int(records[int(index)]["user"] in PAIRED_USERS[fold])
            for index in indices
        ],
        dtype=np.int64,
    )


def raw_metrics(probabilities, indices, labels, fold, records):
    prediction = probabilities.argmax(axis=1)
    domain = domains(indices, fold, records)
    domain_accuracy = {
        str(value): float((prediction[domain == value] == labels[domain == value]).mean())
        for value in range(4)
    }
    return {
        "correct": int((prediction == labels).sum()),
        "rows": len(labels),
        "accuracy": float((prediction == labels).mean()),
        "nll": float(-np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1)).mean()),
        "domain_accuracy": domain_accuracy,
        "test_domain_weighted_accuracy": sum(
            DOMAIN_WEIGHTS[value] * domain_accuracy[str(value)] for value in DOMAIN_WEIGHTS
        ),
    }


def pair_metrics(probabilities, indices, labels, fold, records):
    result = run(fold, (probabilities, indices, labels), records, PAIR_SETTINGS)
    user_domain = {
        user: int(int(user) > 15) + 2 * int(int(user) in PAIRED_USERS[fold])
        for user in result["per_user"]
    }
    domain_accuracy = {}
    for domain in range(4):
        selected = [values for user, values in result["per_user"].items() if user_domain[user] == domain]
        domain_accuracy[str(domain)] = sum(values["correct"] for values in selected) / sum(
            values["rows"] for values in selected
        )
    return {
        "correct": result["correct"],
        "rows": len(labels),
        "accuracy": result["accuracy"],
        "domain_accuracy": domain_accuracy,
        "test_domain_weighted_accuracy": sum(
            DOMAIN_WEIGHTS[value] * domain_accuracy[str(value)] for value in DOMAIN_WEIGHTS
        ),
    }


def main():
    records = get_records()
    folds = []
    for fold in (0, 1):
        control = load(
            ROOT
            / f"experiments/scene_context_v13/outputs/checkpoints/"
            f"v23_shared5_joint_warm_maskdistill_f{fold}/validation.npz"
        )
        candidate = load(
            HERE
            / f"outputs/checkpoints/v84_residual_invariant_skeleton_f{fold}/validation.npz"
        )
        candidate_probabilities = align(control, candidate)
        control_probabilities = control["probabilities"].astype(np.float64)
        variants = {
            "control": control_probabilities,
            "residual_invariant": candidate_probabilities,
            "equal_probability_ensemble": 0.5 * control_probabilities + 0.5 * candidate_probabilities,
        }
        fold_results = {"fold": fold, "variants": {}}
        for name, probabilities in variants.items():
            fold_results["variants"][name] = {
                "raw": raw_metrics(probabilities, control["indices"], control["labels"], fold, records),
                "pair": pair_metrics(probabilities, control["indices"], control["labels"], fold, records),
            }
        folds.append(fold_results)
    summary = {}
    for name in folds[0]["variants"]:
        summary[name] = {}
        for stage in ("raw", "pair"):
            values = [fold["variants"][name][stage] for fold in folds]
            summary[name][stage] = {
                "correct": sum(value["correct"] for value in values),
                "rows": sum(value["rows"] for value in values),
                "accuracy": sum(value["correct"] for value in values) / sum(value["rows"] for value in values),
                "mean_fold_test_domain_weighted_accuracy": sum(
                    value["test_domain_weighted_accuracy"] for value in values
                ) / len(values),
                "fold_test_domain_weighted_accuracy": [
                    value["test_domain_weighted_accuracy"] for value in values
                ],
            }
    control = summary["control"]["pair"]
    candidate = summary["residual_invariant"]["pair"]
    report = {
        "protocol": "One-variable body-invariant residual Skeleton encoder test on the V23 warm recipe.",
        "folds": folds,
        "summary": summary,
        "promote": (
            candidate["correct"] > control["correct"]
            and all(
                candidate_value > control_value
                for candidate_value, control_value in zip(
                    candidate["fold_test_domain_weighted_accuracy"],
                    control["fold_test_domain_weighted_accuracy"],
                )
            )
        ),
    }
    output = HERE / "outputs/evaluation.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"summary": summary, "promote": report["promote"]}, indent=2))


if __name__ == "__main__":
    main()

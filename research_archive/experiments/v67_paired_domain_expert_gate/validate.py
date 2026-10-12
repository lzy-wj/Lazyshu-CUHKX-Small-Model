#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.research_pairaware_base_blend import PAIRED_USERS
from experiments.scene_context_v13.research_repeat_recovery_decoder import run
from src.data import get_records


WEIGHTS = (0.0, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
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
    return candidate["probabilities"][order].astype(np.float64)


def domains(indices, records, fold):
    return np.asarray(
        [
            int(records[int(index)]["user"] > 15)
            + 2 * int(records[int(index)]["user"] in PAIRED_USERS[fold])
            for index in indices
        ],
        dtype=np.int64,
    )


def evaluate(fold, probabilities, archive, records):
    result = run(
        fold,
        (probabilities, archive["indices"], archive["labels"]),
        records,
        SETTINGS,
    )
    result["raw_correct"] = int((probabilities.argmax(1) == archive["labels"]).sum())
    return result


def main():
    records = get_records()
    folds = []
    for fold in (0, 1):
        candidate = load(
            ROOT / f"experiments/v66_supervised_expert_transformer/outputs/crossfit_f{fold}.npz"
        )
        v57_archive = load(
            ROOT / f"experiments/v57_round_transformer_distillation/outputs/predictions_f{fold}.npz"
        )
        folds.append(
            {
                "candidate": candidate["probabilities"].astype(np.float64),
                "v57": align(candidate, v57_archive),
                "archive": candidate,
                "domains": domains(candidate["indices"], records, fold),
            }
        )
    candidates = []
    for weight in WEIGHTS:
        probabilities = (1 - weight) * folds[0]["v57"] + weight * folds[0]["candidate"]
        result = evaluate(0, probabilities, folds[0]["archive"], records)
        candidates.append(
            {
                "weight": weight,
                "result": result,
                "per_domain_correct": {
                    str(domain): result["per_user"][
                        str(
                            records[
                                int(folds[0]["archive"]["indices"][folds[0]["domains"] == domain][0])
                            ]["user"]
                        )
                    ]["correct"]
                    for domain in range(4)
                },
            }
        )
    selected_weights = {}
    for domain in range(4):
        selected_weights[str(domain)] = max(
            candidates,
            key=lambda item: (item["per_domain_correct"][str(domain)], -item["weight"]),
        )["weight"]
    fold_results = []
    prediction_archives = []
    for fold, values in enumerate(folds):
        probabilities = values["v57"].copy()
        for domain, weight in selected_weights.items():
            selected = values["domains"] == int(domain)
            probabilities[selected] = (
                (1 - weight) * values["v57"][selected]
                + weight * values["candidate"][selected]
            )
        result = evaluate(fold, probabilities, values["archive"], records)
        fold_results.append(result)
        prediction_archives.append(probabilities.astype(np.float32))
        output = HERE / f"outputs/predictions_f{fold}.npz"
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output,
            probabilities=probabilities.astype(np.float32),
            indices=values["archive"]["indices"],
            labels=values["archive"]["labels"],
        )
    correct = fold_results[0]["correct"] + fold_results[1]["correct"]
    report = {
        "protocol": (
            "Per-domain weights selected on Fold 0 decoded accuracy with lower-weight tie break; "
            "the four weights are transferred unchanged to Fold 1."
        ),
        "settings": SETTINGS,
        "selected_weights": selected_weights,
        "fold0": fold_results[0],
        "fold1": fold_results[1],
        "combined": {"correct": correct, "rows": 1357, "accuracy": correct / 1357},
        "reference_v59": {"correct": 1197, "rows": 1357},
        "promote": (
            correct > 1197
            and fold_results[0]["correct"] >= 569
            and fold_results[1]["correct"] >= 628
        ),
        "candidates": candidates,
    }
    output = HERE / "outputs/validation.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "candidates"}, indent=2))


if __name__ == "__main__":
    main()


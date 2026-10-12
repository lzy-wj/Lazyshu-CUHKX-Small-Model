#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.research_repeat_recovery_decoder import run
from experiments.v66_supervised_expert_transformer.run import (
    BLEND_WEIGHTS,
    SETTINGS,
    cross_user_selection,
    fold_domains,
    load,
    load_fold,
    raw_metrics,
    train_predict,
)
from src.data import get_records
from src.scratch_protocol import modality_presence


MODALITIES = ("scene_depth", "thermal", "scene_ir", "skeleton", "imu", "radar")


def add_film(values, fold):
    archive = load(
        ROOT
        / f"experiments/v55_modality_film/outputs/checkpoints/"
        f"v55_modality_film_f{fold}/validation.npz"
    )
    positions = {int(index): position for position, index in enumerate(archive["indices"])}
    order = np.asarray([positions[int(index)] for index in values["indices"]])
    values["experts"].append(archive["probabilities"][order].astype(np.float32))
    return values


def decoded(fold, probabilities, values, records):
    return run(
        fold,
        (probabilities, values["indices"], values["labels"]),
        records,
        SETTINGS,
    )


def main():
    records = get_records()
    presence = modality_presence(MODALITIES)
    fold0 = add_film(load_fold(0, presence), 0)
    fold1 = add_film(load_fold(1, presence), 1)
    domains0 = fold_domains(fold0, records, 0)
    domains1 = fold_domains(fold1, records, 1)
    selected, candidates = cross_user_selection(fold0, domains0, records)
    retention = selected["retention_weight"]
    _, prediction1, history01, sequence1 = train_predict(
        fold0, domains0, fold1, domains1, records, retention, 6901
    )
    blind_fold1 = {
        "raw": raw_metrics(prediction1, fold1["labels"]),
        "decoded": decoded(1, prediction1, fold1, records),
    }
    print(json.dumps({"blind_fold1": blind_fold1}, indent=2), flush=True)
    _, prediction0, history10, sequence0 = train_predict(
        fold1, domains1, fold0, domains0, records, retention, 6910
    )
    predictions = (prediction0, prediction1)
    values = (fold0, fold1)
    domains = (domains0, domains1)
    v57 = []
    for fold in (0, 1):
        archive = load(
            ROOT / f"experiments/v57_round_transformer_distillation/outputs/predictions_f{fold}.npz"
        )
        positions = {int(index): position for position, index in enumerate(archive["indices"])}
        order = np.asarray([positions[int(index)] for index in values[fold]["indices"]])
        v57.append(archive["probabilities"][order].astype(np.float32))
    weight_results = []
    for weight in BLEND_WEIGHTS:
        per_fold = []
        for fold in (0, 1):
            probabilities = (1 - weight) * v57[fold] + weight * predictions[fold]
            per_fold.append(
                {
                    "raw": raw_metrics(probabilities, values[fold]["labels"]),
                    "decoded": decoded(fold, probabilities, values[fold], records),
                }
            )
        weight_results.append({"weight": weight, "fold0": per_fold[0], "fold1": per_fold[1]})
    selected_weights = {}
    for domain in range(4):
        selected_user = records[int(fold0["indices"][domains0 == domain][0])]["user"]
        selected_weights[str(domain)] = max(
            weight_results,
            key=lambda item: (
                item["fold0"]["decoded"]["per_user"][str(selected_user)]["correct"],
                -item["weight"],
            ),
        )["weight"]
    gated = []
    for fold in (0, 1):
        probabilities = v57[fold].copy()
        for domain, weight in selected_weights.items():
            selected_rows = domains[fold] == int(domain)
            probabilities[selected_rows] = (
                (1 - weight) * v57[fold][selected_rows]
                + weight * predictions[fold][selected_rows]
            )
        gated.append(
            {
                "raw": raw_metrics(probabilities, values[fold]["labels"]),
                "decoded": decoded(fold, probabilities, values[fold], records),
            }
        )
        output = HERE / f"outputs/crossfit_gated_f{fold}.npz"
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output,
            probabilities=probabilities,
            indices=values[fold]["indices"],
            labels=values[fold]["labels"],
        )
    correct = gated[0]["decoded"]["correct"] + gated[1]["decoded"]["correct"]
    report = {
        "protocol": (
            "Fold-0 LOSO retention selection, one blind Fold-1 evaluation, reverse cross-fit, "
            "and Fold-0 per-domain V57 gate selection transferred unchanged to Fold 1."
        ),
        "selection": selected,
        "selection_candidates": candidates,
        "blind_fold1_standalone": blind_fold1,
        "selected_weights": selected_weights,
        "fold0": gated[0],
        "fold1": gated[1],
        "combined": {"correct": correct, "rows": 1357, "accuracy": correct / 1357},
        "reference_v59": {"correct": 1197, "rows": 1357},
        "promote": (
            correct > 1197
            and gated[0]["decoded"]["correct"] >= 569
            and gated[1]["decoded"]["correct"] >= 628
        ),
        "weight_results": weight_results,
        "training": {"fold0_to_fold1": history01, "fold1_to_fold0": history10},
        "sequences": {"fold0": sequence0, "fold1": sequence1},
    }
    output = HERE / "outputs/validation.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in ("selection_candidates", "weight_results", "training")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


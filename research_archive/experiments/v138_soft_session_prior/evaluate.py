#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.research_flexible_pair_decoder import (
    flexible_assignment,
    prepare_fold,
)
from experiments.scene_context_v13.research_paired_script_decoder import PAIR_SOURCES
from experiments.v109_crossfit_round_transformer.train import v104_probabilities
from experiments.v114_paired_person_fusion.evaluate import PAIRINGS, fused_probabilities
from experiments.v125_domain_complement.evaluate import scores
from experiments.v137_session_matched_decoder.evaluate import day, map_sessions, source_sessions
from src.data import FOLDS, get_records


BONUSES = (0.0, 0.1, 0.4, 0.8, 1.2, 2.0, 4.0)


def decode(fold, probabilities, reference, records, bonus):
    prepared = prepare_fold(
        probabilities,
        reference["indices"],
        reference["labels"],
        records,
        PAIR_SOURCES[fold],
        FOLDS[fold],
    )
    prediction = {
        int(index): int(probability.argmax())
        for index, probability in zip(reference["indices"], probabilities)
    }
    for target, source in PAIRINGS[fold].items():
        target_rounds = [item for item in prepared["rounds"] if item["user"] == target]
        source_values = source_sessions(records, source)
        session_mapping = map_sessions(
            {day(item["start"]) for item in target_rounds}, source_values
        )
        all_templates = [
            (source_day, template)
            for source_day, templates in source_values
            for template in templates
        ]
        for item in target_rounds:
            selected = item["free"]
            best_score = item["free_score"]
            expected_day = session_mapping[day(item["start"])][0]
            for source_day, template in all_templates:
                candidate = flexible_assignment(
                    item["values"],
                    template,
                    item["transition"],
                    transition_strength=0.4,
                    template_strength=1.8,
                    gap_penalty=0.8,
                    maximum_edits=5,
                    beam_size=40,
                )
                if candidate is None:
                    continue
                adjusted = candidate[0] + bonus * candidate[3] * (source_day == expected_day)
                if adjusted > best_score:
                    best_score = adjusted
                    selected = candidate[2]
            prediction.update(zip(item["rows"], selected))
    return np.asarray([prediction[int(index)] for index in reference["indices"]])


def fold_values(fold, records, bonus):
    reference, _, v104, _ = v104_probabilities(fold, records)
    probabilities, _ = fused_probabilities(fold, reference, v104, records, 0.6)
    paired = decode(fold, probabilities, reference, records, bonus)
    incumbent = np.load(
        ROOT
        / f"experiments/v127_supervision_routed_system/outputs/v127_validation_f{fold}.npz"
    )["pseudo_labels"]
    users = np.asarray([records[int(index)]["user"] for index in reference["indices"]])
    selected = np.isin(users, list(PAIRINGS[fold]))
    prediction = incumbent.copy()
    prediction[selected] = paired[selected]
    return scores(prediction, reference, fold, records), prediction, incumbent, reference


def main():
    records = get_records()
    grid = []
    for bonus in BONUSES:
        metrics, prediction, incumbent, reference = fold_values(0, records, bonus)
        labels = reference["labels"]
        grid.append(
            {
                "bonus": bonus,
                "fold0": metrics,
                "changes_vs_v127": int((prediction != incumbent).sum()),
                "gains": int(((incumbent != labels) & (prediction == labels)).sum()),
                "losses": int(((incumbent == labels) & (prediction != labels)).sum()),
            }
        )
    selected = max(
        grid,
        key=lambda item: (
            item["fold0"]["correct"],
            item["fold0"]["test_domain_weighted_accuracy"],
            -item["bonus"],
        ),
    )
    fold1, prediction, incumbent, reference = fold_values(1, records, selected["bonus"])
    labels = reference["labels"]
    fold1_deltas = {
        "metrics": fold1,
        "changes_vs_v127": int((prediction != incumbent).sum()),
        "gains": int(((incumbent != labels) & (prediction == labels)).sum()),
        "losses": int(((incumbent == labels) & (prediction != labels)).sum()),
    }
    correct = selected["fold0"]["correct"] + fold1["correct"]
    rows = selected["fold0"]["rows"] + fold1["rows"]
    report = {
        "protocol": "Bonus selected on Fold0 and transferred unchanged to Fold1.",
        "bonuses": BONUSES,
        "grid": grid,
        "selected": selected | {"fold1": fold1_deltas},
        "summary": {"correct": correct, "rows": rows, "accuracy": correct / rows},
        "v127": {"correct": 1205, "rows": 1357},
        "promote": correct > 1205,
    }
    output = HERE / "outputs/evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

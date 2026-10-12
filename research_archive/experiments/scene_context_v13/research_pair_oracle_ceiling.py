#!/usr/bin/env python3
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.research_flexible_pair_decoder import prepare_fold
from experiments.scene_context_v13.research_pairaware_base_blend import PAIRED_USERS
from experiments.scene_context_v13.research_paired_script_decoder import PAIR_SOURCES
from experiments.scene_context_v13.research_sequence_decoder import selected_probabilities
from src.data import FOLDS, get_records


def lcs_length(left, right):
    previous = [0] * (len(right) + 1)
    for left_value in left:
        current = [0]
        for position, right_value in enumerate(right, start=1):
            current.append(
                previous[position - 1] + 1
                if left_value == right_value
                else max(previous[position], current[-1])
            )
        previous = current
    return previous[-1]


def edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for left_position, left_value in enumerate(left, start=1):
        current = [left_position]
        for right_position, right_value in enumerate(right, start=1):
            current.append(
                min(
                    previous[right_position] + 1,
                    current[-1] + 1,
                    previous[right_position - 1] + int(left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def main():
    records = get_records()
    reports = []
    for fold in (0, 1):
        probabilities, indices, labels = selected_probabilities(fold)
        prepared = prepare_fold(
            probabilities,
            indices,
            labels,
            records,
            PAIR_SOURCES[fold],
            FOLDS[fold],
        )
        by_user = {}
        for user in sorted(PAIRED_USERS[fold]):
            source = PAIR_SOURCES[fold][int(user > 15)][0]
            templates = prepared["templates"][source]
            user_rounds = [item for item in prepared["rounds"] if item["user"] == user]
            rows = 0
            matched = 0
            exact = 0
            within_five = 0
            distances = []
            for item in user_rounds:
                truth = tuple(records[index]["label"] for index in item["rows"])
                rows += len(truth)
                matched += max(lcs_length(template, truth) for template in templates)
                distance = min(edit_distance(template, truth) for template in templates)
                distances.append(distance)
                exact += int(distance == 0)
                within_five += int(distance <= 5)
            by_user[str(user)] = {
                "source_user": source,
                "rounds": len(user_rounds),
                "rows": rows,
                "maximum_source_matched_rows": matched,
                "source_coverage": matched / rows,
                "exact_template_rounds": exact,
                "rounds_within_five_edits": within_five,
                "mean_minimum_edit_distance": sum(distances) / len(distances),
                "maximum_minimum_edit_distance": max(distances),
            }
        reports.append({"fold": fold, "users": by_user})
    paired_rows = sum(item["rows"] for report in reports for item in report["users"].values())
    matched_rows = sum(
        item["maximum_source_matched_rows"]
        for report in reports
        for item in report["users"].values()
    )
    report = {
        "protocol": (
            "Label-aware ceiling audit only; no setting is selected and no test prediction is produced. "
            "Each held-out paired round is compared with every source-user script by LCS and edit distance."
        ),
        "folds": reports,
        "combined": {
            "paired_rows": paired_rows,
            "maximum_source_matched_rows": matched_rows,
            "source_coverage": matched_rows / paired_rows,
        },
    }
    output = ROOT / "experiments/scene_context_v13/outputs/pair_oracle_ceiling.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

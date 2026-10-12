#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.research_flexible_pair_decoder import flexible_assignment, prepare_fold
from experiments.scene_context_v13.research_paired_script_decoder import PAIR_SOURCES, user_templates
from experiments.v114_paired_person_fusion.evaluate import PAIRINGS
from experiments.v125_domain_complement.evaluate import scores
from experiments.v135_end_to_end_label_experts.evaluate import components
from experiments.v84_residual_invariant_skeleton.evaluate import pair_metrics
from scripts.evaluate_repeat_decoder import rounds
from src.data import FOLDS, get_records


SKIP_PENALTIES = (0.4, 0.8, 1.2, 1.6, 2.0)
MATCH_MARGIN = 0.0


def source_rounds(records, user):
    indices = [
        record["index"]
        for record in records
        if record["split"] == "train" and record["user"] == user
    ]
    output = []
    for item in rounds(indices, records):
        output.append(
            {
                "rows": item["rows"],
                "labels": tuple(records[index]["label"] for index in item["rows"]),
            }
        )
    return output


def align_rounds(target_rounds, source_rounds, skip_penalty):
    target_count = len(target_rounds)
    source_count = len(source_rounds)
    values = np.full((target_count, source_count), -1e9, dtype=np.float64)
    candidates = {}
    for target_position, target in enumerate(target_rounds):
        for source_position, source in enumerate(source_rounds):
            candidate = flexible_assignment(
                target["values"],
                source["labels"],
                target["transition"],
                transition_strength=0.4,
                template_strength=1.8,
                gap_penalty=0.8,
                maximum_edits=5,
                beam_size=40,
            )
            if candidate is not None:
                values[target_position, source_position] = candidate[0] - target["free_score"]
                candidates[(target_position, source_position)] = candidate
    dp = np.full((target_count + 1, source_count + 1), -np.inf, dtype=np.float64)
    move = np.zeros((target_count + 1, source_count + 1), dtype=np.int8)
    dp[0, 0] = 0
    for target_position in range(target_count + 1):
        for source_position in range(source_count + 1):
            if target_position == 0 and source_position == 0:
                continue
            choices = []
            if target_position and source_position:
                choices.append((dp[target_position - 1, source_position - 1] + values[target_position - 1, source_position - 1], 0))
            if target_position:
                choices.append((dp[target_position - 1, source_position] - skip_penalty, 1))
            if source_position:
                choices.append((dp[target_position, source_position - 1] - skip_penalty, 2))
            dp[target_position, source_position], move[target_position, source_position] = max(choices)
    target_position, source_position = target_count, source_count
    matches = []
    while target_position or source_position:
        direction = move[target_position, source_position]
        if direction == 0:
            matches.append((target_position - 1, source_position - 1))
            target_position -= 1
            source_position -= 1
        elif direction == 1:
            target_position -= 1
        else:
            source_position -= 1
    return values, candidates, matches[::-1], float(dp[-1, -1])


def decode_fold(fold, reference, probabilities, records, skip_penalty):
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
    reports = []
    for target, source in PAIRINGS[fold].items():
        target_values = [item for item in prepared["rounds"] if item["user"] == target]
        source_values = source_rounds(records, source)
        values, candidates, matches, score = align_rounds(target_values, source_values, skip_penalty)
        changed = selected = 0
        for target_position, source_position in matches:
            candidate = candidates.get((target_position, source_position))
            if candidate is None or values[target_position, source_position] <= MATCH_MARGIN:
                continue
            item = target_values[target_position]
            selected_sequence = candidate[2]
            changed += sum(left != right for left, right in zip(item["free"], selected_sequence))
            selected += 1
            prediction.update(zip(item["rows"], selected_sequence))
        reports.append(
            {
                "target": target,
                "source": source,
                "target_rounds": len(target_values),
                "source_rounds": len(source_values),
                "aligned_pairs": len(matches),
                "selected_rounds": selected,
                "changed_predictions": changed,
                "alignment_score": score,
            }
        )
    return np.asarray([prediction[int(index)] for index in reference["indices"]]), reports


def main():
    records = get_records()
    folds = {}
    for fold in (0, 1):
        reference, _, v114, _ = components(fold, records)
        incumbent = np.load(
            ROOT / f"experiments/v127_supervision_routed_system/outputs/v127_validation_f{fold}.npz"
        )["pseudo_labels"]
        folds[fold] = (reference, v114, incumbent)
    candidates = []
    for skip_penalty in SKIP_PENALTIES:
        reference, v114, incumbent = folds[0]
        prediction, reports = decode_fold(0, reference, v114, records, skip_penalty)
        users = np.asarray([records[int(index)]["user"] for index in reference["indices"]])
        paired = np.isin(users, list(PAIRINGS[0]))
        merged = incumbent.copy()
        merged[paired] = prediction[paired]
        candidates.append(
            {
                "skip_penalty": skip_penalty,
                "fold0": scores(merged, reference, 0, records),
                "reports": reports,
            }
        )
    selected = max(
        candidates,
        key=lambda item: (
            item["fold0"]["correct"],
            item["fold0"]["test_domain_weighted_accuracy"],
            -item["skip_penalty"],
        ),
    )
    reference, v114, incumbent = folds[1]
    prediction, reports = decode_fold(1, reference, v114, records, selected["skip_penalty"])
    users = np.asarray([records[int(index)]["user"] for index in reference["indices"]])
    paired = np.isin(users, list(PAIRINGS[1]))
    merged = incumbent.copy()
    merged[paired] = prediction[paired]
    fold1 = scores(merged, reference, 1, records)
    correct = selected["fold0"]["correct"] + fold1["correct"]
    report = {
        "protocol": "Monotonic target-to-source round alignment; skip penalty selected on Fold0 only.",
        "candidates": candidates,
        "selected": selected | {"fold1": fold1, "fold1_reports": reports},
        "summary": {"correct": correct, "rows": 1357, "accuracy": correct / 1357},
        "v127": {"correct": 1205, "rows": 1357},
        "promote": correct > 1205 and selected["fold0"]["correct"] >= 569 and fold1["correct"] >= 636,
    }
    output = HERE / "outputs/evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selected": report["selected"], "summary": report["summary"], "promote": report["promote"]}, indent=2))


if __name__ == "__main__":
    main()

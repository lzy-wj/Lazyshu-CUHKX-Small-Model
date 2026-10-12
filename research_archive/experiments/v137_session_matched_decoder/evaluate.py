#!/usr/bin/env python3
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
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
from scripts.evaluate_repeat_decoder import rounds
from src.data import FOLDS, get_records


def day(timestamp):
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")


def source_sessions(records, user):
    indices = [
        record["index"]
        for record in records
        if record["split"] == "train" and record["user"] == user
    ]
    grouped = defaultdict(lambda: defaultdict(list))
    starts = {}
    for item in rounds(indices, records):
        session = day(item["start"])
        prefix = Counter(
            "-".join(records[index]["trial"].split("-")[:2]) for index in item["rows"]
        ).most_common(1)[0][0]
        grouped[session][prefix].append(
            tuple(records[index]["label"] for index in item["rows"])
        )
        starts[session] = min(starts.get(session, item["start"]), item["start"])
    output = []
    for session in sorted(grouped, key=starts.get):
        templates = [
            Counter(values).most_common(1)[0][0]
            for values in grouped[session].values()
        ]
        output.append((session, templates))
    return output


def map_sessions(target_sessions, source_values):
    target_sessions = sorted(target_sessions)
    if len(target_sessions) == 1:
        source_positions = [len(source_values) // 2]
    else:
        source_positions = np.linspace(
            0, len(source_values) - 1, len(target_sessions)
        ).round().astype(int).tolist()
    return {
        target: source_values[position]
        for target, position in zip(target_sessions, source_positions)
    }


def decode(fold, probabilities, reference, records):
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
    reports = {}
    for target, source in PAIRINGS[fold].items():
        target_rounds = [item for item in prepared["rounds"] if item["user"] == target]
        target_sessions = {day(item["start"]) for item in target_rounds}
        mapping = map_sessions(target_sessions, source_sessions(records, source))
        changed = selected_rounds = 0
        for item in target_rounds:
            selected = item["free"]
            best = (item["free_score"], item["free_score"], selected, 0, 0)
            source_day, templates = mapping[day(item["start"])]
            for template in templates:
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
                if candidate is not None and candidate[0] > best[0]:
                    best = candidate
            if best[2] != selected:
                selected_rounds += 1
                changed += sum(left != right for left, right in zip(selected, best[2]))
                selected = best[2]
            prediction.update(zip(item["rows"], selected))
        reports[str(target)] = {
            "source": source,
            "session_mapping": {
                target_day: source_day for target_day, (source_day, _) in mapping.items()
            },
            "rounds": len(target_rounds),
            "selected_rounds": selected_rounds,
            "changed_predictions": changed,
        }
    ordered = np.asarray([prediction[int(index)] for index in reference["indices"]])
    return ordered, reports


def main():
    records = get_records()
    results = []
    for fold in (0, 1):
        reference, _, v104, _ = v104_probabilities(fold, records)
        probabilities, _ = fused_probabilities(fold, reference, v104, records, 0.6)
        session_prediction, reports = decode(fold, probabilities, reference, records)
        incumbent = np.load(
            ROOT
            / f"experiments/v127_supervision_routed_system/outputs/v127_validation_f{fold}.npz"
        )["pseudo_labels"]
        paired_users = set(PAIRINGS[fold])
        users = np.asarray([records[int(index)]["user"] for index in reference["indices"]])
        prediction = incumbent.copy()
        selected = np.isin(users, list(paired_users))
        prediction[selected] = session_prediction[selected]
        labels = reference["labels"]
        changed = prediction != incumbent
        results.append(
            {
                "fold": fold,
                "metrics": scores(prediction, reference, fold, records),
                "changes_vs_v127": int(changed.sum()),
                "gains": int(((incumbent != labels) & (prediction == labels)).sum()),
                "losses": int(((incumbent == labels) & (prediction != labels)).sum()),
                "users": reports,
            }
        )
    correct = sum(item["metrics"]["correct"] for item in results)
    rows = sum(item["metrics"]["rows"] for item in results)
    report = {
        "protocol": (
            "Relative chronological session matching has no fitted parameter; "
            "the identical rule is evaluated on both held-out folds."
        ),
        "folds": results,
        "summary": {"correct": correct, "rows": rows, "accuracy": correct / rows},
        "v127": {"correct": 1205, "rows": 1357},
        "promote": correct > 1205
        and all(item["metrics"]["correct"] >= baseline for item, baseline in zip(results, (569, 636))),
    }
    output = HERE / "outputs/evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

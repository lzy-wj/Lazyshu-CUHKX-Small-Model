#!/usr/bin/env python3
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.research_flexible_pair_decoder import (
    flexible_assignment,
    prepare_fold,
)
from experiments.scene_context_v13.research_paired_script_decoder import (
    PAIR_SOURCES,
    user_templates,
)
from experiments.v114_paired_person_fusion.evaluate import PAIRINGS
from experiments.v125_domain_complement.evaluate import scores
from experiments.v135_end_to_end_label_experts.evaluate import components
from experiments.v84_residual_invariant_skeleton.evaluate import PAIR_SETTINGS
from src.data import FOLDS, get_records


def target_units(prepared, target):
    grouped = defaultdict(list)
    for position, item in enumerate(prepared["rounds"]):
        if item["user"] != target:
            continue
        key = ("round", position) if item["group"] is None else ("group", item["group"])
        grouped[key].append(item)
    return list(grouped.values())


def source_templates(records, source):
    return [
        sequence
        for sequences in user_templates(records, source).values()
        for sequence in sequences
    ]


def decode_user(prepared, target, source, records):
    units = target_units(prepared, target)
    templates = source_templates(records, source)
    candidates = {}
    gains = np.full((len(units), len(templates) + len(units)), -1e6, dtype=np.float64)
    gains[:, len(templates):] = 0.0
    for unit_position, unit in enumerate(units):
        item = unit[0]
        for template_position, template in enumerate(templates):
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
            gain = (candidate[0] - item["free_score"]) * len(unit)
            gains[unit_position, template_position] = gain
            candidates[(unit_position, template_position)] = candidate
    rows, columns = linear_sum_assignment(-gains)
    assignment = dict(zip(rows.tolist(), columns.tolist()))
    predictions = {}
    selected_units = changed = 0
    for unit_position, unit in enumerate(units):
        template_position = assignment[unit_position]
        selected = unit[0]["free"]
        if template_position < len(templates) and gains[unit_position, template_position] > 0:
            selected = candidates[(unit_position, template_position)][2]
            selected_units += 1
        for item in unit:
            changed += sum(left != right for left, right in zip(item["free"], selected))
            predictions.update(zip(item["rows"], selected))
    return predictions, {
        "target": target,
        "source": source,
        "units": len(units),
        "templates": len(templates),
        "selected_units": selected_units,
        "changed_predictions": changed,
    }


def decode(fold, reference, probabilities, records):
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
        values, report = decode_user(prepared, target, source, records)
        prediction.update(values)
        reports.append(report)
    ordered = np.asarray(
        [prediction[int(index)] for index in reference["indices"]], dtype=np.int64
    )
    return ordered, reports


def main():
    records = get_records()
    results = []
    for fold in (0, 1):
        reference, v71, v114, _ = components(fold, records)
        incumbent = np.load(
            ROOT
            / f"experiments/v127_supervision_routed_system/outputs/v127_validation_f{fold}.npz"
        )["pseudo_labels"]
        matched, reports = decode(fold, reference, v114, records)
        users = np.asarray(
            [records[int(index)]["user"] for index in reference["indices"]]
        )
        paired = np.isin(users, list(PAIRINGS[fold]))
        prediction = incumbent.copy()
        prediction[paired] = matched[paired]
        labels = reference["labels"]
        changed = prediction != incumbent
        results.append(
            {
                "fold": fold,
                "metrics": scores(prediction, reference, fold, records),
                "changes": int(changed.sum()),
                "gains": int(((incumbent != labels) & (prediction == labels)).sum()),
                "losses": int(((incumbent == labels) & (prediction != labels)).sum()),
                "users": reports,
            }
        )
    correct = sum(item["metrics"]["correct"] for item in results)
    rows = sum(item["metrics"]["rows"] for item in results)
    report = {
        "protocol": (
            "Fixed V127 strengths with globally one-to-one source-template use; "
            "the identical parameter-free rule is applied to both folds."
        ),
        "folds": results,
        "summary": {"correct": correct, "rows": rows, "accuracy": correct / rows},
        "v127": {"correct": 1205, "rows": 1357},
        "promote": correct > 1205
        and all(
            item["metrics"]["correct"] >= baseline
            for item, baseline in zip(results, (569, 636))
        ),
    }
    output = HERE / "outputs/evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()


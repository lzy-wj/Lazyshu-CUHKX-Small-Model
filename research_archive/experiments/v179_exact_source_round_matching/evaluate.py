#!/usr/bin/env python3
import json
import sys
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
from experiments.scene_context_v13.research_paired_script_decoder import PAIR_SOURCES
from experiments.v114_paired_person_fusion.evaluate import PAIRINGS
from experiments.v125_domain_complement.evaluate import scores
from experiments.v135_end_to_end_label_experts.evaluate import components
from scripts.evaluate_repeat_decoder import rounds
from src.data import FOLDS, get_records


TEMPLATE_STRENGTHS = (0.5, 0.8, 1.2, 1.8)
GAP_PENALTIES = (0.3, 0.8)
MAXIMUM_EDITS = (0, 1, 3, 5)
MINIMUM_MARGINS = (0.0, 0.5, 1.0, 2.0)


def source_rounds(records, user):
    indices = [
        record["index"]
        for record in records
        if record["split"] == "train" and record["user"] == user
    ]
    return [
        tuple(records[index]["label"] for index in item["rows"])
        for item in rounds(indices, records)
    ]


def decode_user(
    prepared,
    target,
    source,
    records,
    template_strength,
    gap_penalty,
    maximum_edits,
    minimum_margin,
    cache,
):
    target_rounds = [item for item in prepared["rounds"] if item["user"] == target]
    templates = source_rounds(records, source)
    cache_key = (target, source, template_strength, gap_penalty, maximum_edits)
    if cache_key not in cache:
        gains = np.full(
            (len(target_rounds), len(templates) + len(target_rounds)),
            -1e9,
            dtype=np.float64,
        )
        gains[:, len(templates):] = 0.0
        candidates = {}
        for target_position, item in enumerate(target_rounds):
            for template_position, template in enumerate(templates):
                candidate = flexible_assignment(
                    item["values"],
                    template,
                    item["transition"],
                    transition_strength=0.4,
                    template_strength=template_strength,
                    gap_penalty=gap_penalty,
                    maximum_edits=maximum_edits,
                    beam_size=40,
                )
                if candidate is None:
                    continue
                gains[target_position, template_position] = candidate[0] - item["free_score"]
                candidates[(target_position, template_position)] = candidate
        row_positions, column_positions = linear_sum_assignment(-gains)
        assignment = dict(zip(row_positions.tolist(), column_positions.tolist()))
        cache[cache_key] = gains, candidates, assignment
    gains, candidates, assignment = cache[cache_key]
    predictions = {}
    selected_rounds = 0
    changed_predictions = 0
    selected_margins = []
    for target_position, item in enumerate(target_rounds):
        template_position = assignment[target_position]
        selected = item["free"]
        if (
            template_position < len(templates)
            and gains[target_position, template_position] >= minimum_margin
        ):
            selected = candidates[(target_position, template_position)][2]
            selected_rounds += 1
            selected_margins.append(float(gains[target_position, template_position]))
        changed_predictions += sum(
            left != right for left, right in zip(item["free"], selected)
        )
        predictions.update(zip(item["rows"], selected))
    return predictions, {
        "target": target,
        "source": source,
        "target_rounds": len(target_rounds),
        "source_rounds": len(templates),
        "selected_rounds": selected_rounds,
        "changed_from_free": changed_predictions,
        "minimum_selected_margin": min(selected_margins, default=None),
        "mean_selected_margin": float(np.mean(selected_margins)) if selected_margins else None,
    }


def evaluate_setting(fold, values, records, setting, cache):
    reference, prepared, incumbent = values
    candidate_by_index = {}
    user_reports = []
    for target, source in PAIRINGS[fold].items():
        predictions, report = decode_user(
            prepared,
            target,
            source,
            records,
            **setting,
            cache=cache,
        )
        candidate_by_index.update(predictions)
        user_reports.append(report)

    users = np.asarray(
        [records[int(index)]["user"] for index in reference["indices"]],
        dtype=np.int64,
    )
    paired = np.isin(users, list(PAIRINGS[fold]))
    candidate = np.asarray(
        [
            candidate_by_index.get(int(index), int(prediction))
            for index, prediction in zip(reference["indices"], incumbent)
        ],
        dtype=np.int64,
    )
    merged = incumbent.copy()
    merged[paired] = candidate[paired]
    labels = reference["labels"]
    changed = merged != incumbent
    return {
        "metrics": scores(merged, reference, fold, records),
        "changes": int(changed.sum()),
        "gains": int(((incumbent != labels) & (merged == labels)).sum()),
        "losses": int(((incumbent == labels) & (merged != labels)).sum()),
        "neutral": int((changed & (incumbent != labels) & (merged != labels)).sum()),
        "users": user_reports,
    }


def main():
    records = get_records()
    folds = {}
    for fold in (0, 1):
        reference, _, v114, _ = components(fold, records)
        incumbent = np.load(
            ROOT
            / "experiments/v127_supervision_routed_system/outputs"
            / f"v127_validation_f{fold}.npz"
        )["pseudo_labels"].astype(np.int64)
        prepared = prepare_fold(
            v114,
            reference["indices"],
            reference["labels"],
            records,
            PAIR_SOURCES[fold],
            FOLDS[fold],
        )
        folds[fold] = (reference, prepared, incumbent)

    grid = []
    fold0_cache = {}
    for template_strength in TEMPLATE_STRENGTHS:
        for gap_penalty in GAP_PENALTIES:
            for maximum_edits in MAXIMUM_EDITS:
                for minimum_margin in MINIMUM_MARGINS:
                    setting = {
                        "template_strength": template_strength,
                        "gap_penalty": gap_penalty,
                        "maximum_edits": maximum_edits,
                        "minimum_margin": minimum_margin,
                    }
                    fold0 = evaluate_setting(
                        0, folds[0], records, setting, fold0_cache
                    )
                    grid.append(setting | {"fold0": fold0})

    selected = max(
        grid,
        key=lambda item: (
            item["fold0"]["metrics"]["correct"],
            -item["fold0"]["losses"],
            -item["fold0"]["changes"],
            -item["template_strength"],
        ),
    )
    setting = {key: selected[key] for key in (
        "template_strength",
        "gap_penalty",
        "maximum_edits",
        "minimum_margin",
    )}
    fold1 = evaluate_setting(1, folds[1], records, setting, {})
    correct = selected["fold0"]["metrics"]["correct"] + fold1["metrics"]["correct"]
    report = {
        "protocol": (
            "Each actual source recording is retained as a separate labeled template and can be "
            "assigned to at most one target round. Settings are selected on Fold0; Fold1 is blind."
        ),
        "grid": grid,
        "selected": setting | {"fold0": selected["fold0"], "fold1": fold1},
        "summary": {"correct": correct, "rows": 1357, "accuracy": correct / 1357},
        "v127": {"correct": 1205, "rows": 1357},
        "promote": (
            correct > 1205
            and selected["fold0"]["metrics"]["correct"] >= 569
            and fold1["metrics"]["correct"] >= 636
        ),
    }
    output = HERE / "outputs/evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "selected": report["selected"],
        "summary": report["summary"],
        "promote": report["promote"],
    }, indent=2))


if __name__ == "__main__":
    main()

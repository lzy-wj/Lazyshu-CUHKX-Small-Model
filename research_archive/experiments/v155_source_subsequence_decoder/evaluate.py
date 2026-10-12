#!/usr/bin/env python3
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.research_flexible_pair_decoder import model_score, prepare_fold
from experiments.scene_context_v13.research_paired_script_decoder import PAIR_SOURCES, user_templates
from experiments.v114_paired_person_fusion.evaluate import PAIRINGS
from experiments.v125_domain_complement.evaluate import scores
from experiments.v135_end_to_end_label_experts.evaluate import components
from src.data import FOLDS, get_records


REPEATS = 3
TEMPLATE_STRENGTH = 1.8
GAP_PENALTY = 0.8


def source_templates(records, source):
    return [
        sequence
        for sequences in user_templates(records, source).values()
        for sequence in sequences
    ]


def subsequence_candidate(item, template):
    target_length = len(item["rows"])
    if target_length > len(template):
        return None
    unary = np.log(np.clip(item["values"], 1e-12, 1))
    best = None
    for positions in itertools.combinations(range(len(template)), target_length):
        sequence = tuple(int(template[position]) for position in positions)
        likelihood = model_score(sequence, unary, item["transition"], 0.4)
        augmented = (
            likelihood
            + TEMPLATE_STRENGTH * target_length
            - GAP_PENALTY * (len(template) - target_length)
        )
        candidate = (augmented, likelihood, sequence)
        if best is None or candidate[0] > best[0]:
            best = candidate
    return best


def decode_user(prepared, target, source, records):
    target_rounds = [item for item in prepared["rounds"] if item["user"] == target]
    unique_templates = source_templates(records, source)
    templates = [template for template in unique_templates for _ in range(REPEATS)]
    gains = np.full(
        (len(target_rounds), len(templates) + len(target_rounds)),
        -1e6,
        dtype=np.float64,
    )
    gains[:, len(templates):] = 0
    candidates = {}
    for target_position, item in enumerate(target_rounds):
        for template_position, template in enumerate(templates):
            candidate = subsequence_candidate(item, template)
            if candidate is None:
                continue
            gains[target_position, template_position] = candidate[0] - item["free_score"]
            candidates[(target_position, template_position)] = candidate
    rows, columns = linear_sum_assignment(-gains)
    assignment = dict(zip(rows.tolist(), columns.tolist()))
    predictions = {}
    selected_rounds = changed = 0
    usage = np.zeros(len(unique_templates), dtype=np.int64)
    for target_position, item in enumerate(target_rounds):
        template_position = assignment[target_position]
        selected = item["free"]
        if template_position < len(templates) and gains[target_position, template_position] > 0:
            selected = candidates[(target_position, template_position)][2]
            usage[template_position // REPEATS] += 1
            selected_rounds += 1
        changed += sum(left != right for left, right in zip(item["free"], selected))
        predictions.update(zip(item["rows"], selected))
    return predictions, {
        "target": target,
        "source": source,
        "rounds": len(target_rounds),
        "templates": len(unique_templates),
        "selected_rounds": selected_rounds,
        "changed_predictions": changed,
        "template_usage": usage.tolist(),
    }


def decode_fold(fold, reference, probabilities, records):
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
    return np.asarray([prediction[int(index)] for index in reference["indices"]]), reports


def main():
    records = get_records()
    results = []
    for fold in (0, 1):
        reference, _, v114, _ = components(fold, records)
        incumbent = np.load(
            ROOT / f"experiments/v127_supervision_routed_system/outputs/v127_validation_f{fold}.npz"
        )["pseudo_labels"]
        matched, reports = decode_fold(fold, reference, v114, records)
        users = np.asarray([records[int(index)]["user"] for index in reference["indices"]])
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
        "protocol": "Fixed deletion-only paired-source subsequences with three slots per script.",
        "folds": results,
        "summary": {"correct": correct, "rows": rows, "accuracy": correct / rows},
        "v127": {"correct": 1205, "rows": 1357},
        "promote": correct > 1205 and all(
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

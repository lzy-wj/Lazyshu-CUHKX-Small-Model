#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.research_flexible_pair_decoder import flexible_assignment, model_score
from experiments.scene_context_v13.research_paired_script_decoder import user_templates
from experiments.scene_context_v13.research_sequence_decoder import repeat_pool
from scripts.evaluate_repeat_decoder import assign_transition_unique, transition_matrix
from src.data import get_records


SOURCE_BY_DATE = {
    "2025-05-31": 20,
    "2025-06-01": 20,
    "2025-06-02": 21,
    "2025-06-12": None,
    "2025-06-13": None,
    "2025-06-16": 16,
    "2025-06-17": 16,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "outputs/predictions/final_legal_multimodal_v12_lowrank.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/predictions/v13_pair_aware.npz",
    )
    parser.add_argument(
        "--submission",
        type=Path,
        default=ROOT / "outputs/submissions/v13_pair_aware.csv",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "experiments/scene_context_v13/outputs/pair_aware_test.json",
    )
    parser.add_argument("--maximum-edits", type=int, default=5)
    parser.add_argument("--gap-penalty", type=float, default=0.8)
    parser.add_argument("--template-strength", type=float, default=1.8)
    parser.add_argument("--alignment-beam-size", type=int, default=40)
    return parser.parse_args()


def main():
    args = parse_args()
    records = get_records()
    archive = np.load(args.input)
    probabilities = archive["raw_probabilities"] if "raw_probabilities" in archive else archive["probabilities"]
    indices = archive["indices"]
    if len(indices) != sum(record["split"] == "test" for record in records):
        raise ValueError("Prediction archive does not cover the full test set")
    probability_by_index, round_values, groups = repeat_pool(probabilities, indices, records)
    transitions = [transition_matrix(records, set(), environment=value) for value in (0, 1)]
    source_users = {value for value in SOURCE_BY_DATE.values() if value is not None}
    templates = {
        user: sorted({sequence for values in user_templates(records, user).values() for sequence in values})
        for user in source_users
    }
    decoded = {int(index): probability.copy() for index, probability in zip(indices, probabilities)}
    changed_by_date = Counter()
    rounds_by_date = Counter()
    selected_by_date = Counter()
    margin_by_date = defaultdict(float)
    for item in round_values:
        date = datetime.fromtimestamp(item["start"]).strftime("%Y-%m-%d")
        if date not in SOURCE_BY_DATE:
            raise ValueError(f"Unexpected test acquisition date: {date}")
        environment = int(date >= "2025-06-09")
        transition = transitions[environment]
        values = np.stack([probability_by_index[index] for index in item["rows"]])
        free = assign_transition_unique(values, transition, 0.4, beam_size=100).argmax(axis=1)
        unary = np.log(np.clip(values, 1e-12, 1.0))
        free_score = model_score(free, unary, transition, 0.4)
        selected = tuple(int(value) for value in free)
        source = SOURCE_BY_DATE[date]
        best = (free_score, free_score, selected, 0, 0)
        if source is not None:
            for template in templates[source]:
                candidate = flexible_assignment(
                    values,
                    template,
                    transition,
                    transition_strength=0.4,
                    template_strength=args.template_strength,
                    gap_penalty=args.gap_penalty,
                    maximum_edits=args.maximum_edits,
                    beam_size=args.alignment_beam_size,
                )
                if candidate is not None and candidate[0] > best[0]:
                    best = candidate
        rounds_by_date[date] += 1
        margin_by_date[date] += float(best[0] - free_score)
        if best[2] != selected:
            selected_by_date[date] += 1
            changed_by_date[date] += sum(left != right for left, right in zip(best[2], selected))
            selected = best[2]
        constrained = np.full_like(values, 1e-12)
        constrained[np.arange(len(selected)), selected] = 1.0
        for position, index in enumerate(item["rows"]):
            decoded[index] = constrained[position]
    ordered = np.stack([decoded[int(index)] for index in indices])
    reference = archive["probabilities"] if "probabilities" in archive else probabilities
    predictions = ordered.argmax(axis=1)
    reference_predictions = reference.argmax(axis=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, probabilities=ordered, raw_probabilities=probabilities, indices=indices)
    args.submission.parent.mkdir(parents=True, exist_ok=True)
    with args.submission.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "prediction"])
        for index, prediction in zip(indices, predictions):
            writer.writerow([records[int(index)]["id"], int(prediction)])
    report = {
        "input": str(args.input.resolve().relative_to(ROOT)),
        "output": str(args.output.resolve().relative_to(ROOT)),
        "submission": str(args.submission.resolve().relative_to(ROOT)),
        "protocol": {
            "maximum_edits": args.maximum_edits,
            "gap_penalty": args.gap_penalty,
            "template_strength": args.template_strength,
            "alignment_beam_size": args.alignment_beam_size,
            "transition_strength": 0.4,
            "source_by_date": SOURCE_BY_DATE,
            "date_mapping_basis": "Acquisition order and strict template-margin audit; no test labels.",
        },
        "rounds": len(round_values),
        "repeat_groups": len(groups),
        "changed_vs_input_decoder": int((predictions != reference_predictions).sum()),
        "prediction_distribution": {str(key): value for key, value in sorted(Counter(predictions.tolist()).items())},
        "per_date": {
            date: {
                "source_user": SOURCE_BY_DATE[date],
                "rounds": rounds_by_date[date],
                "selected_rounds": selected_by_date[date],
                "changed_predictions": changed_by_date[date],
                "aggregate_margin": margin_by_date[date],
            }
            for date in sorted(rounds_by_date)
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

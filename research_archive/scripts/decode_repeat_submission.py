#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_repeat_decoder import (
    assign_transition_unique,
    assign_unique,
    cooccurrence_matrix,
    probability_similarity,
    pool_repeat_probabilities,
    record_interval,
    transition_matrix,
)
from src.data import ROOT, get_records


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-weight", type=float, default=0.3)
    parser.add_argument("--candidate-temperature", type=float, default=1.5)
    parser.add_argument("--auxiliary", type=Path)
    parser.add_argument("--auxiliary-weight", type=float, default=0.0)
    parser.add_argument("--auxiliary-temperature", type=float, default=1.0)
    parser.add_argument("--auxiliary-maximum-base-confidence", type=float)
    parser.add_argument("--maximum-gap", type=float, default=240)
    parser.add_argument("--minimum-similarity", type=float, default=0.7)
    parser.add_argument("--cooccurrence-strength", type=float, default=0.0)
    parser.add_argument("--transition-strength", type=float, default=0.0)
    parser.add_argument("--transition-scope", choices=("global", "environment"), default="global")
    parser.add_argument("--repeat-pool-power", type=float, default=1.0)
    parser.add_argument("--name", required=True)
    return parser.parse_args()


def load_aligned(path, indices):
    values = np.load(path)
    positions = {int(index): position for position, index in enumerate(values["indices"])}
    if set(positions) != set(indices):
        raise ValueError(f"Prediction indices do not match test records: {path}")
    return values["probabilities"][[positions[index] for index in indices]]


def temperature_scale(probabilities, temperature):
    values = np.log(np.clip(probabilities, 1e-12, 1)) / temperature
    values -= values.max(axis=1, keepdims=True)
    values = np.exp(values)
    return values / values.sum(axis=1, keepdims=True)


def segment_rounds(indices, records, gap_seconds=30):
    values = [(record_interval(records[index]), index) for index in indices]
    missing = [index for interval, index in values if interval is None]
    values = [(interval, index) for interval, index in values if interval is not None]
    values.sort()
    output = []
    for interval, index in values:
        if not output or interval[0] - output[-1][-1][0][1] > gap_seconds:
            output.append([])
        output[-1].append((interval, index))
    rounds = [
        {
            "rows": [index for _, index in group],
            "start": group[0][0][0],
            "end": max(interval[1] for interval, _ in group),
        }
        for group in output
    ]
    return rounds, missing


def find_repeat_groups(round_values, probability_by_index, maximum_gap, minimum_similarity):
    groups = []
    position = 0
    while position < len(round_values):
        triple = round_values[position:position + 3]
        if len(triple) == 3:
            gaps = [triple[index + 1]["start"] - triple[index]["end"] for index in range(2)]
            similarities = [
                probability_similarity(triple[0], triple[index], probability_by_index) for index in (1, 2)
            ]
            if max(gaps) <= maximum_gap and min(similarities) >= minimum_similarity:
                groups.append(triple)
                position += 3
                continue
        pair = round_values[position:position + 2]
        if len(pair) == 2:
            gap = pair[1]["start"] - pair[0]["end"]
            similarity = probability_similarity(pair[0], pair[1], probability_by_index)
            if gap <= maximum_gap and similarity >= minimum_similarity:
                groups.append(pair)
                position += 2
                continue
        position += 1
    return groups


def main():
    args = parse_args()
    gated_auxiliary = args.auxiliary_maximum_base_confidence is not None
    if (not 0 <= args.candidate_weight <= 1 or not 0 <= args.auxiliary_weight <= 1
            or (not gated_auxiliary and args.candidate_weight + args.auxiliary_weight > 1)
            or args.candidate_temperature <= 0 or args.auxiliary_temperature <= 0):
        raise ValueError("Invalid blend settings")
    if gated_auxiliary and not 0 <= args.auxiliary_maximum_base_confidence <= 1:
        raise ValueError("Auxiliary confidence threshold must be in [0, 1]")
    records = get_records()
    indices = [record["index"] for record in records if record["split"] == "test"]
    base = load_aligned(args.base, indices)
    candidate = load_aligned(args.candidate, indices)
    auxiliary = load_aligned(args.auxiliary, indices) if args.auxiliary else None
    raw = ((1 - args.candidate_weight) * base
           + args.candidate_weight * temperature_scale(candidate, args.candidate_temperature))
    auxiliary_rows = np.zeros(len(raw), dtype=bool)
    if auxiliary is not None:
        auxiliary_probability = temperature_scale(auxiliary, args.auxiliary_temperature)
        if gated_auxiliary:
            auxiliary_rows = raw.max(axis=1) < args.auxiliary_maximum_base_confidence
            raw[auxiliary_rows] = (
                (1 - args.auxiliary_weight) * raw[auxiliary_rows]
                + args.auxiliary_weight * auxiliary_probability[auxiliary_rows]
            )
        else:
            raw = ((1 - args.auxiliary_weight) * raw
                   + args.auxiliary_weight * auxiliary_probability)
    elif args.auxiliary_weight:
        raise ValueError("An auxiliary weight requires --auxiliary")
    probability_by_index = {index: probability.copy() for index, probability in zip(indices, raw)}
    round_values, missing = segment_rounds(indices, records)
    groups = find_repeat_groups(
        round_values, probability_by_index, args.maximum_gap, args.minimum_similarity
    )
    decoded = {index: probability.copy() for index, probability in probability_by_index.items()}
    for group in groups:
        pooled = pool_repeat_probabilities(
            np.stack([[probability_by_index[index] for index in item["rows"]] for item in group]),
            args.repeat_pool_power,
        )
        for item in group:
            for row, index in enumerate(item["rows"]):
                decoded[index] = pooled[row]
    cooccurrence = cooccurrence_matrix(records, set())
    transition = transition_matrix(records, set())
    environment_transitions = [transition_matrix(records, set(), environment=value) for value in (0, 1)]
    environment_centers = []
    for environment in (0, 1):
        starts = [record_interval(record)[0] for record in records
                  if record["split"] == "train" and (record["user"] > 15) == bool(environment)
                  and record_interval(record) is not None]
        environment_centers.append(float(np.median(starts)))
    environment_round_counts = [0, 0]
    for item in round_values:
        probabilities = np.stack([decoded[index] for index in item["rows"]])
        environment = int(np.argmin([abs(item["start"] - center) for center in environment_centers]))
        environment_round_counts[environment] += 1
        selected_transition = environment_transitions[environment] if args.transition_scope == "environment" else transition
        constrained = (
            assign_transition_unique(probabilities, selected_transition, args.transition_strength)
            if args.transition_strength > 0
            else assign_unique(probabilities, cooccurrence, args.cooccurrence_strength)
        )
        for row, index in enumerate(item["rows"]):
            decoded[index] = constrained[row]
    decoded_probabilities = np.stack([decoded[index] for index in indices])
    raw_labels = raw.argmax(1)
    decoded_labels = decoded_probabilities.argmax(1)
    test_csv = pd.read_csv(
        ROOT / "data/downloads/CUHK-X_Small_Model_Track/Small-Model-Track/Testing/test_file/test.csv"
    )
    prediction_table = pd.DataFrame(
        {"path": [records[index]["id"] for index in indices], "prediction": decoded_labels}
    )
    submission = test_csv[["path"]].merge(prediction_table, on="path", how="left")
    if submission["prediction"].isna().any() or len(submission) != 405:
        raise ValueError("Submission paths do not match test.csv")
    submission["prediction"] = submission["prediction"].astype(int)
    submission_path = ROOT / "outputs/submissions" / f"{args.name}.csv"
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_path, index=False)
    output_path = ROOT / "outputs/predictions" / f"{args.name}.npz"
    np.savez(
        output_path,
        probabilities=decoded_probabilities,
        raw_probabilities=raw,
        indices=np.asarray(indices),
    )
    report = {
        "base": str(args.base),
        "candidate": str(args.candidate),
        "candidate_weight": args.candidate_weight,
        "candidate_temperature": args.candidate_temperature,
        "auxiliary": str(args.auxiliary) if args.auxiliary else None,
        "auxiliary_weight": args.auxiliary_weight,
        "auxiliary_temperature": args.auxiliary_temperature,
        "auxiliary_maximum_base_confidence": args.auxiliary_maximum_base_confidence,
        "auxiliary_rows": int(auxiliary_rows.sum()),
        "maximum_gap": args.maximum_gap,
        "minimum_similarity": args.minimum_similarity,
        "cooccurrence_strength": args.cooccurrence_strength,
        "transition_strength": args.transition_strength,
        "transition_scope": args.transition_scope,
        "repeat_pool_power": args.repeat_pool_power,
        "environment_round_counts": environment_round_counts,
        "rounds": len(round_values),
        "repeat_groups": len(groups),
        "grouped_records": sum(len(item["rows"]) for group in groups for item in group),
        "missing_timestamp_records": missing,
        "changed_predictions": int((raw_labels != decoded_labels).sum()),
        "raw_class_counts": np.bincount(raw_labels, minlength=40).tolist(),
        "decoded_class_counts": np.bincount(decoded_labels, minlength=40).tolist(),
        "submission": str(submission_path),
        "note": "Uses only model probabilities and raw acquisition intervals; no test labels, user IDs, trial IDs, or timestamp-to-label matching.",
    }
    output_path.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

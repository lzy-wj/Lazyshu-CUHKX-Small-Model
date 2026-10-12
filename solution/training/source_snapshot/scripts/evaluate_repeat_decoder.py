#!/usr/bin/env python3
import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import FOLDS


TIMESTAMP_PATTERN = re.compile(r"(20\d\d-\d\d-\d\d_\d\d-\d\d-\d\d\.\d+)")
INTERVAL_CACHE = {}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold0", type=Path, required=True)
    parser.add_argument("--fold1", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=Path("data/metadata/samples.json"))
    return parser.parse_args()


def record_interval(record):
    if record["index"] in INTERVAL_CACHE:
        return INTERVAL_CACHE[record["index"]]
    for modality in ("Depth_Color", "IR", "Radar"):
        folder = Path(record["modalities"].get(modality, ""))
        if not folder.is_dir():
            continue
        values = []
        for path in folder.iterdir():
            match = TIMESTAMP_PATTERN.search(path.name)
            if match:
                values.append(datetime.strptime(match.group(1), "%Y-%m-%d_%H-%M-%S.%f").timestamp())
        if values:
            INTERVAL_CACHE[record["index"]] = (min(values), max(values))
            return INTERVAL_CACHE[record["index"]]
    imu_folder = Path(record["modalities"].get("IMU", ""))
    values = []
    if imu_folder.is_dir():
        for path in imu_folder.glob("*.csv"):
            try:
                with path.open(encoding="utf-8-sig") as handle:
                    rows = csv.reader(handle)
                    next(rows, None)
                    for row in rows:
                        if row:
                            values.append(datetime.fromisoformat(row[0]).timestamp())
            except (OSError, UnicodeDecodeError, ValueError):
                continue
    if values:
        INTERVAL_CACHE[record["index"]] = (min(values), max(values))
        return INTERVAL_CACHE[record["index"]]
    INTERVAL_CACHE[record["index"]] = None
    return None


def cooccurrence_matrix(records, excluded_users):
    grouped = defaultdict(set)
    for record in records:
        if record["split"] == "train" and record["user"] not in excluded_users:
            grouped[(record["user"], record["trial"])].add(record["label"])
    frequencies = np.ones(40, dtype=np.float64)
    counts = np.ones((40, 40), dtype=np.float64)
    for labels in grouped.values():
        labels = sorted(labels)
        frequencies[labels] += 1
        counts[np.ix_(labels, labels)] += 1
    total = len(grouped) + 1
    pointwise_information = np.log(counts * total / np.outer(frequencies, frequencies))
    pointwise_information = np.clip(pointwise_information, -2, 2)
    np.fill_diagonal(pointwise_information, 0)
    return pointwise_information


def transition_matrix(records, excluded_users, smoothing=0.5, environment=None):
    indices = [record["index"] for record in records
               if record["split"] == "train" and record["user"] not in excluded_users
               and (environment is None or (record["user"] > 15) == bool(environment))]
    transitions = np.full((40, 40), smoothing, dtype=np.float64)
    frequencies = np.full(40, smoothing, dtype=np.float64)
    for user in sorted({records[index]["user"] for index in indices}):
        user_indices = [index for index in indices if records[index]["user"] == user]
        for item in rounds(user_indices, records):
            labels = [records[index]["label"] for index in item["rows"]]
            for label in labels:
                frequencies[label] += 1
            for left, right in zip(labels, labels[1:]):
                transitions[left, right] += 1
    background = frequencies / frequencies.sum()
    conditional = transitions / transitions.sum(axis=1, keepdims=True)
    return np.clip(np.log(conditional / background[None]), -4, 4)


def assign_unique(probabilities, cooccurrence, strength):
    unary = np.log(np.clip(probabilities, 1e-12, 1))
    rows, labels = linear_sum_assignment(-unary)
    assignment = np.empty(len(probabilities), dtype=np.int64)
    assignment[rows] = labels
    if strength > 0:
        for _ in range(3):
            scores = unary.copy()
            for row in range(len(probabilities)):
                context = np.delete(assignment, row)
                if len(context):
                    scores[row] += strength * cooccurrence[:, context].mean(axis=1)
            rows, labels = linear_sum_assignment(-scores)
            assignment[rows] = labels
    constrained = np.full_like(probabilities, 1e-12)
    constrained[np.arange(len(probabilities)), assignment] = 1
    return constrained


def assign_transition_unique(probabilities, transition, strength, beam_size=100,
                             position_scores=None, position_strength=0):
    unary = np.log(np.clip(probabilities, 1e-12, 1))
    if position_scores is not None:
        if position_scores.shape != unary.shape:
            raise ValueError("Position-prior shape does not match round probabilities")
        unary = unary + position_strength * position_scores
    beams = [(0.0, (), 0)]
    for position in range(len(probabilities)):
        candidates = []
        for score, sequence, used in beams:
            values = unary[position] if position == 0 else unary[position] + strength * transition[sequence[-1]]
            for label in np.argsort(values)[::-1]:
                label = int(label)
                if not (used >> label) & 1:
                    candidates.append((score + values[label], sequence + (label,), used | (1 << label)))
        candidates.sort(key=lambda value: value[0], reverse=True)
        beams = candidates[:beam_size]
    assignment = np.asarray(beams[0][1], dtype=np.int64)
    constrained = np.full_like(probabilities, 1e-12)
    constrained[np.arange(len(probabilities)), assignment] = 1
    return constrained


def rounds(indices, records, gap_seconds=30):
    values = [(record_interval(records[int(index)]), int(index)) for index in indices]
    values = [value for value in values if value[0] is not None]
    values.sort()
    output = []
    for interval, index in values:
        if not output or interval[0] - output[-1][-1][0][1] > gap_seconds:
            output.append([])
        output[-1].append((interval, index))
    return [
        {
            "rows": [index for _, index in group],
            "start": group[0][0][0],
            "end": max(interval[1] for interval, _ in group),
        }
        for group in output
    ]


def probability_similarity(left, right, probability_by_index):
    if len(left["rows"]) != len(right["rows"]):
        return -1.0
    first = np.stack([probability_by_index[index] for index in left["rows"]])
    second = np.stack([probability_by_index[index] for index in right["rows"]])
    first = np.sqrt(np.clip(first, 1e-12, 1))
    second = np.sqrt(np.clip(second, 1e-12, 1))
    return float((first * second).sum(axis=1).mean())


def pool_repeat_probabilities(probabilities, power=1.0, confidence_power=0.0):
    if power <= 0:
        raise ValueError("Repeat pooling power must be positive")
    weights = np.ones((*probabilities.shape[:2], 1), dtype=np.float64)
    if confidence_power > 0:
        weights = np.power(probabilities.max(axis=2, keepdims=True), confidence_power)
    weights = weights / weights.sum(axis=0, keepdims=True)
    if power == 1:
        return np.sum(weights * probabilities, axis=0)
    pooled = np.sum(weights * np.power(probabilities, power), axis=0) ** (1 / power)
    return pooled / pooled.sum(axis=1, keepdims=True)


def repeat_groups(round_values, probability_by_index, maximum_gap, minimum_similarity):
    output = []
    position = 0
    while position < len(round_values):
        candidate = round_values[position:position + 3]
        if len(candidate) == 3:
            gaps = [candidate[i + 1]["start"] - candidate[i]["end"] for i in range(2)]
            similarities = [
                probability_similarity(candidate[0], candidate[i], probability_by_index) for i in (1, 2)
            ]
            if max(gaps) <= maximum_gap and min(similarities) >= minimum_similarity:
                output.append(candidate)
                position += 3
                continue
        candidate = round_values[position:position + 2]
        if len(candidate) == 2:
            gap = candidate[1]["start"] - candidate[0]["end"]
            similarity = probability_similarity(candidate[0], candidate[1], probability_by_index)
            if gap <= maximum_gap and similarity >= minimum_similarity:
                output.append(candidate)
                position += 2
                continue
        position += 1
    return output


def decode(probabilities, indices, records, maximum_gap, minimum_similarity, enforce_unique=False,
           cooccurrence=None, cooccurrence_strength=0, transition=None, transition_strength=0,
           environment_transitions=None, repeat_pool_power=1.0, beam_size=100,
           repeat_confidence_power=0.0, position_priors=None, position_strength=0.0):
    probability_by_index = {int(index): probability for index, probability in zip(indices, probabilities)}
    round_values = rounds(indices, records)
    groups = repeat_groups(round_values, probability_by_index, maximum_gap, minimum_similarity)
    decoded = {int(index): probability.copy() for index, probability in zip(indices, probabilities)}
    grouped_records = 0
    pure_groups = 0
    for group in groups:
        size = len(group[0]["rows"])
        pooled = pool_repeat_probabilities(
            np.stack([[probability_by_index[index] for index in item["rows"]] for item in group]),
            repeat_pool_power,
            repeat_confidence_power,
        )
        for item in group:
            for row, index in enumerate(item["rows"]):
                decoded[index] = pooled[row]
                grouped_records += 1
        prefixes = {
            (records[index]["user"], "-".join(records[index]["trial"].split("-")[:2]))
            for item in group for index in item["rows"]
        }
        pure_groups += len(prefixes) == 1
    if enforce_unique:
        for item in round_values:
            round_probabilities = np.stack([decoded[index] for index in item["rows"]])
            selected_transition = transition
            if environment_transitions is not None:
                environment = int(records[item["rows"][0]]["user"] > 15)
                selected_transition = environment_transitions[environment]
            selected_position_scores = None
            if position_priors is not None:
                environment = int(records[item["rows"][0]]["user"] > 15)
                selected_position_scores = position_priors[environment].get(len(item["rows"]))
            constrained = (
                assign_transition_unique(
                    round_probabilities, selected_transition, transition_strength, beam_size=beam_size,
                    position_scores=selected_position_scores, position_strength=position_strength,
                )
                if transition_strength > 0
                else assign_unique(round_probabilities, cooccurrence, cooccurrence_strength)
            )
            for row, index in enumerate(item["rows"]):
                decoded[index] = constrained[row]
    ordered = np.stack([decoded[int(index)] for index in indices])
    labels = np.asarray([records[int(index)]["label"] for index in indices])
    return {
        "accuracy": float((ordered.argmax(axis=1) == labels).mean()),
        "groups": len(groups),
        "pure_groups": pure_groups,
        "grouped_records": grouped_records,
        "rounds": len(round_values),
    }


def true_group_upper_bound(probabilities, indices, records, enforce_unique=False):
    probability_by_index = {int(index): probability for index, probability in zip(indices, probabilities)}
    grouped = defaultdict(list)
    for item in rounds(indices, records):
        first = records[item["rows"][0]]
        prefix = "-".join(first["trial"].split("-")[:2])
        grouped[(first["user"], prefix)].append(item)
    decoded = {int(index): probability.copy() for index, probability in zip(indices, probabilities)}
    covered = 0
    for group in grouped.values():
        if len(group) < 2 or len({len(item["rows"]) for item in group}) != 1:
            continue
        pooled = np.mean(
            np.stack([[probability_by_index[index] for index in item["rows"]] for item in group]), axis=0
        )
        if enforce_unique:
            row_assignment, labels = linear_sum_assignment(-np.log(np.clip(pooled, 1e-12, 1)))
            constrained = np.full_like(pooled, 1e-12)
            constrained[row_assignment, labels] = 1
            pooled = constrained
        for item in group:
            for row, index in enumerate(item["rows"]):
                decoded[index] = pooled[row]
                covered += 1
    labels = np.asarray([records[int(index)]["label"] for index in indices])
    ordered = np.stack([decoded[int(index)] for index in indices])
    return {"accuracy": float((ordered.argmax(1) == labels).mean()), "covered": covered}


def load(path):
    values = np.load(path)
    return values["probabilities"], values["indices"]


def main():
    args = parse_args()
    records = json.loads(args.metadata.read_text())
    folds = [load(args.fold0), load(args.fold1)]
    matrices = [cooccurrence_matrix(records, set(FOLDS[fold])) for fold in (0, 1)]
    transitions = [transition_matrix(records, set(FOLDS[fold])) for fold in (0, 1)]
    environment_transitions = [
        [transition_matrix(records, set(FOLDS[fold]), environment=environment) for environment in (0, 1)]
        for fold in (0, 1)
    ]
    grids = []
    for maximum_gap in (90, 120, 150, 180, 240, 360):
        for minimum_similarity in (0.40, 0.50, 0.60, 0.70, 0.80):
            for strength in (0, 0.1, 0.2, 0.3, 0.5, 0.8):
                result = decode(
                    *folds[0], records, maximum_gap, minimum_similarity, enforce_unique=True,
                    cooccurrence=matrices[0], cooccurrence_strength=strength,
                )
                grids.append((result["accuracy"], maximum_gap, minimum_similarity, strength, result))
            for strength in (0.1, 0.2, 0.3, 0.5, 0.8):
                result = decode(
                    *folds[0], records, maximum_gap, minimum_similarity, enforce_unique=True,
                    transition=transitions[0], transition_strength=strength,
                )
                grids.append((result["accuracy"], maximum_gap, minimum_similarity, 0, result, strength))
            for strength in (0.1, 0.2, 0.3, 0.4, 0.5):
                result = decode(
                    *folds[0], records, maximum_gap, minimum_similarity, enforce_unique=True,
                    transition_strength=strength, environment_transitions=environment_transitions[0],
                )
                grids.append((result["accuracy"], maximum_gap, minimum_similarity, 0, result, strength, "environment"))
    selected = max(grids, key=lambda value: (value[0], -value[1], value[2], -value[3], -len(value)))
    transition_strength = selected[5] if len(selected) > 5 else 0
    transition_scope = selected[6] if len(selected) > 6 else "global" if transition_strength else "none"
    report = {
        "selection": {
            "fold": 0,
            "maximum_gap": selected[1],
            "minimum_similarity": selected[2],
            "cooccurrence_strength": selected[3],
            "transition_strength": transition_strength,
            "transition_scope": transition_scope,
            "result": selected[4],
        },
        "folds": [],
    }
    for fold, values in enumerate(folds):
        probabilities, indices = values
        labels = np.asarray([records[int(index)]["label"] for index in indices])
        report["folds"].append(
            {
                "fold": fold,
                "raw_accuracy": float((probabilities.argmax(1) == labels).mean()),
                "blind_repeat": decode(
                    probabilities, indices, records, selected[1], selected[2], enforce_unique=False
                ),
                "blind_repeat_unique": decode(
                    probabilities, indices, records, selected[1], selected[2], enforce_unique=True,
                    cooccurrence=matrices[fold], cooccurrence_strength=selected[3],
                    transition=transitions[fold], transition_strength=transition_strength,
                    environment_transitions=(environment_transitions[fold]
                                             if transition_scope == "environment" else None),
                ),
                "true_group_upper_bound": true_group_upper_bound(
                    probabilities, indices, records, enforce_unique=False
                ),
                "true_group_unique_upper_bound": true_group_upper_bound(
                    probabilities, indices, records, enforce_unique=True
                ),
            }
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

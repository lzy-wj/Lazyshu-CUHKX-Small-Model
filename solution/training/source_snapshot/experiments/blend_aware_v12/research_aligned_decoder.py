#!/usr/bin/env python3
import json
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_repeat_decoder import assign_transition_unique, rounds, transition_matrix
from src.data import FOLDS, get_records


def load(path):
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key] for key in values.files}


def align(reference, candidate):
    positions = {int(index): position for position, index in enumerate(candidate["indices"])}
    order = np.asarray([positions[int(index)] for index in reference["indices"]])
    if "labels" in candidate and not np.array_equal(reference["labels"], candidate["labels"][order]):
        raise ValueError("Prediction labels do not align")
    return candidate["probabilities"][order].astype(np.float64)


def temperature_scale(probabilities, temperature):
    logits = np.log(np.clip(probabilities, 1e-12, 1)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def final_probabilities(fold):
    base = load(ROOT / f"outputs/study_r2plus1d/lightweight_base_int8/base_fold{fold}.npz")
    candidate = load(
        ROOT / "experiments/blend_aware_v12/outputs/predictions"
        / f"v12_mixup_f{fold}_selected_raw_int8.npz"
    )
    auxiliary = load(ROOT / f"outputs/study_x3dl_soup/selected_cropblend50_fold{fold}.npz")
    candidate_probabilities = align(base, candidate)
    auxiliary_probabilities = align(base, auxiliary)
    probabilities = 0.4 * base["probabilities"] + 0.6 * temperature_scale(candidate_probabilities, 2.0)
    gated = probabilities.max(axis=1) < 0.15
    probabilities[gated] = 0.9 * probabilities[gated] + 0.1 * auxiliary_probabilities[gated]
    return probabilities, base["indices"].astype(np.int64), base["labels"].astype(np.int64)


def sequence_alignment(first, second, gap_penalty):
    similarity = np.sqrt(np.clip(first, 1e-12, 1)) @ np.sqrt(np.clip(second, 1e-12, 1)).T
    rows, columns = similarity.shape
    scores = np.full((rows + 1, columns + 1), -np.inf)
    moves = np.zeros((rows + 1, columns + 1), dtype=np.int8)
    scores[0, 0] = 0
    scores[1:, 0] = -gap_penalty * np.arange(1, rows + 1)
    scores[0, 1:] = -gap_penalty * np.arange(1, columns + 1)
    moves[1:, 0] = 1
    moves[0, 1:] = 2
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            choices = (
                scores[row - 1, column - 1] + similarity[row - 1, column - 1],
                scores[row - 1, column] - gap_penalty,
                scores[row, column - 1] - gap_penalty,
            )
            move = int(np.argmax(choices))
            scores[row, column] = choices[move]
            moves[row, column] = move
    matched = []
    row, column = rows, columns
    while row or column:
        move = int(moves[row, column])
        if move == 0:
            matched.append((row - 1, column - 1, float(similarity[row - 1, column - 1])))
            row -= 1
            column -= 1
        elif move == 1:
            row -= 1
        else:
            column -= 1
    return matched[::-1]


def match_rounds(first, second, probability_by_index, minimum_similarity, minimum_coverage,
                 maximum_length_difference, gap_penalty):
    if abs(len(first["rows"]) - len(second["rows"])) > maximum_length_difference:
        return None
    first_probabilities = np.stack([probability_by_index[index] for index in first["rows"]])
    second_probabilities = np.stack([probability_by_index[index] for index in second["rows"]])
    if len(first["rows"]) == len(second["rows"]):
        similarity = np.sqrt(np.clip(first_probabilities, 1e-12, 1) * np.clip(second_probabilities, 1e-12, 1)).sum(1)
        matched = [(position, position, float(value)) for position, value in enumerate(similarity)]
    else:
        matched = sequence_alignment(first_probabilities, second_probabilities, gap_penalty)
    coverage = len(matched) / max(len(first["rows"]), len(second["rows"]))
    mean_similarity = float(np.mean([value for _, _, value in matched])) if matched else -1
    if coverage < minimum_coverage or mean_similarity < minimum_similarity:
        return None
    return matched


def aligned_repeat_groups(round_values, probability_by_index, maximum_gap, minimum_similarity,
                          minimum_coverage, maximum_length_difference, gap_penalty):
    output = []
    position = 0
    while position < len(round_values):
        candidate = round_values[position:position + 3]
        if len(candidate) == 3:
            gaps = [candidate[index + 1]["start"] - candidate[index]["end"] for index in range(2)]
            matches = [
                match_rounds(
                    candidate[0], candidate[index], probability_by_index, minimum_similarity,
                    minimum_coverage, maximum_length_difference, gap_penalty,
                )
                for index in (1, 2)
            ]
            if max(gaps) <= maximum_gap and all(match is not None for match in matches):
                output.append((candidate, matches))
                position += 3
                continue
        candidate = round_values[position:position + 2]
        if len(candidate) == 2:
            gap = candidate[1]["start"] - candidate[0]["end"]
            matched = match_rounds(
                candidate[0], candidate[1], probability_by_index, minimum_similarity,
                minimum_coverage, maximum_length_difference, gap_penalty,
            )
            if gap <= maximum_gap and matched is not None:
                output.append((candidate, [matched]))
                position += 2
                continue
        position += 1
    return output


def power_pool(probabilities, power):
    if power == 1:
        return probabilities.mean(axis=0)
    pooled = np.mean(np.power(probabilities, power), axis=0) ** (1 / power)
    return pooled / pooled.sum()


def decode_aligned(probabilities, indices, records, transitions, maximum_gap, minimum_similarity,
                   minimum_coverage, maximum_length_difference, gap_penalty, repeat_pool_power,
                   transition_strength=0.4, beam_size=100):
    probability_by_index = {int(index): probability for index, probability in zip(indices, probabilities)}
    round_values = rounds(indices, records)
    groups = aligned_repeat_groups(
        round_values, probability_by_index, maximum_gap, minimum_similarity, minimum_coverage,
        maximum_length_difference, gap_penalty,
    )
    decoded = {index: probability.copy() for index, probability in probability_by_index.items()}
    aligned_records = 0
    unequal_groups = 0
    for group, matches in groups:
        mappings = [{first: second for first, second, _ in matched} for matched in matches]
        unequal_groups += len({len(item["rows"]) for item in group}) > 1
        for reference_position, reference_index in enumerate(group[0]["rows"]):
            matched_indices = [reference_index]
            for item, mapping in zip(group[1:], mappings):
                if reference_position in mapping:
                    matched_indices.append(item["rows"][mapping[reference_position]])
            if len(matched_indices) < 2:
                continue
            pooled = power_pool(np.stack([probability_by_index[index] for index in matched_indices]), repeat_pool_power)
            for index in matched_indices:
                decoded[index] = pooled
                aligned_records += 1
    for item in round_values:
        round_probabilities = np.stack([decoded[index] for index in item["rows"]])
        environment = int(records[item["rows"][0]]["user"] > 15)
        constrained = assign_transition_unique(
            round_probabilities, transitions[environment], transition_strength, beam_size=beam_size
        )
        for row, index in enumerate(item["rows"]):
            decoded[index] = constrained[row]
    ordered = np.stack([decoded[int(index)] for index in indices])
    labels = np.asarray([records[int(index)]["label"] for index in indices])
    return {
        "accuracy": float((ordered.argmax(1) == labels).mean()),
        "correct": int((ordered.argmax(1) == labels).sum()),
        "groups": len(groups),
        "unequal_groups": int(unequal_groups),
        "aligned_records": aligned_records,
    }


def main():
    started = time.monotonic()
    records = get_records()
    folds = [final_probabilities(fold) for fold in (0, 1)]
    transitions = [
        [transition_matrix(records, set(FOLDS[fold]), environment=environment) for environment in (0, 1)]
        for fold in (0, 1)
    ]
    candidates = []
    for minimum_similarity in (0.65, 0.7, 0.75, 0.8):
        for minimum_coverage in (0.6, 0.75, 0.9):
            for maximum_length_difference in (1, 2):
                for gap_penalty in (0.1, 0.25, 0.5):
                    for repeat_pool_power in (1.0, 1.5, 2.0):
                        result = decode_aligned(
                            folds[0][0], folds[0][1], records, transitions[0], 120,
                            minimum_similarity, minimum_coverage, maximum_length_difference,
                            gap_penalty, repeat_pool_power,
                        )
                        candidates.append({
                            "minimum_similarity": minimum_similarity,
                            "minimum_coverage": minimum_coverage,
                            "maximum_length_difference": maximum_length_difference,
                            "gap_penalty": gap_penalty,
                            "repeat_pool_power": repeat_pool_power,
                            "fold0": result,
                        })
    selected = max(
        candidates,
        key=lambda item: (
            item["fold0"]["correct"], -item["fold0"]["unequal_groups"],
            item["minimum_similarity"], item["minimum_coverage"],
        ),
    )
    selected["fold1"] = decode_aligned(
        folds[1][0], folds[1][1], records, transitions[1], 120,
        selected["minimum_similarity"], selected["minimum_coverage"],
        selected["maximum_length_difference"], selected["gap_penalty"],
        selected["repeat_pool_power"],
    )
    report = {
        "protocol": "Aligned repeat decoder selected on fold0 only; fold1 blind; no user/trial IDs used for grouping",
        "search_configurations": len(candidates),
        "selected": selected,
        "combined": {
            "correct": selected["fold0"]["correct"] + selected["fold1"]["correct"],
            "rows": len(folds[0][2]) + len(folds[1][2]),
            "reference_correct": 1100,
        },
        "top_fold0": sorted(candidates, key=lambda item: item["fold0"]["correct"], reverse=True)[:20],
        "elapsed_seconds": time.monotonic() - started,
    }
    report["combined"]["accuracy"] = report["combined"]["correct"] / report["combined"]["rows"]
    report["combined"]["delta_correct"] = report["combined"]["correct"] - report["combined"]["reference_correct"]
    output = ROOT / "experiments/blend_aware_v12/outputs/aligned_repeat_decoder_search.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "top_fold0"}, indent=2))


if __name__ == "__main__":
    main()

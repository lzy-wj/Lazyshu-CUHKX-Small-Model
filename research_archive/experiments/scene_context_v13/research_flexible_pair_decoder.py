#!/usr/bin/env python3
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.research_paired_script_decoder import PAIR_SOURCES, user_templates
from experiments.scene_context_v13.research_sequence_decoder import repeat_pool, selected_probabilities
from experiments.blend_aware_v12.research_aligned_decoder import aligned_repeat_groups, power_pool
from scripts.evaluate_repeat_decoder import assign_transition_unique, cooccurrence_matrix, rounds, transition_matrix
from src.data import FOLDS, get_records


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/scene_context_v13/outputs/flexible_pair_decoder.json",
    )
    return parser.parse_args()


def model_score(sequence, unary, transition, transition_strength):
    positions = np.arange(len(sequence))
    score = float(unary[positions, sequence].sum())
    if len(sequence) > 1:
        score += transition_strength * sum(
            transition[left, right] for left, right in zip(sequence, sequence[1:])
        )
    return score


def flexible_assignment(
    probabilities,
    template,
    transition,
    transition_strength,
    template_strength,
    gap_penalty,
    maximum_edits,
    beam_size=80,
    allow_substitution=False,
    insertion_prior=None,
    insertion_strength=0.0,
):
    unary = np.log(np.clip(probabilities, 1e-12, 1.0))
    target_length = len(probabilities)
    source_length = len(template)
    if abs(target_length - source_length) > maximum_edits:
        return None
    beams = [(0.0, 0.0, (), 0, 0, 0, 0)]
    for position in range(target_length):
        candidates = []
        for augmented, likelihood, sequence, used, source_position, matched, edits in beams:
            remaining_budget = maximum_edits - edits
            maximum_delete = min(remaining_budget, source_length - source_position)
            for deleted in range(maximum_delete + 1):
                aligned_source = source_position + deleted
                deletion_cost = gap_penalty * deleted
                if aligned_source < source_length:
                    label = int(template[aligned_source])
                    if not (used >> label) & 1:
                        transition_value = 0.0 if not sequence else transition_strength * transition[sequence[-1], label]
                        value = float(unary[position, label] + transition_value)
                        candidates.append(
                            (
                                augmented + value + template_strength - deletion_cost,
                                likelihood + value,
                                sequence + (label,),
                                used | (1 << label),
                                aligned_source + 1,
                                matched + 1,
                                edits + deleted,
                            )
                        )
                if edits + deleted < maximum_edits:
                    values = unary[position].copy()
                    if sequence:
                        values += transition_strength * transition[sequence[-1]]
                    if allow_substitution and aligned_source < source_length:
                        source_label = int(template[aligned_source])
                        for label in np.argsort(values)[::-1][:12]:
                            label = int(label)
                            if label == source_label or (used >> label) & 1:
                                continue
                            value = float(values[label])
                            candidates.append(
                                (
                                    augmented + value - deletion_cost - gap_penalty,
                                    likelihood + value,
                                    sequence + (label,),
                                    used | (1 << label),
                                    aligned_source + 1,
                                    matched,
                                    edits + deleted + 1,
                                )
                            )
                    augmented_values = values.copy()
                    if insertion_prior is not None:
                        augmented_values += insertion_strength * insertion_prior
                    for label in np.argsort(augmented_values)[::-1][:12]:
                        label = int(label)
                        if (used >> label) & 1:
                            continue
                        value = float(values[label])
                        candidates.append(
                            (
                                augmented + value + insertion_strength * (
                                    0.0 if insertion_prior is None else insertion_prior[label]
                                ) - deletion_cost - gap_penalty,
                                likelihood + value,
                                sequence + (label,),
                                used | (1 << label),
                                aligned_source,
                                matched,
                                edits + deleted + 1,
                            )
                        )
        candidates.sort(key=lambda item: item[0], reverse=True)
        beams = candidates[:beam_size]
        if not beams:
            return None
    completed = []
    minimum_matches = max(1, math.ceil(min(target_length, source_length) * 0.6))
    for augmented, likelihood, sequence, used, source_position, matched, edits in beams:
        total_edits = edits + source_length - source_position
        if total_edits <= maximum_edits and matched >= minimum_matches:
            completed.append(
                (
                    augmented - gap_penalty * (source_length - source_position),
                    likelihood,
                    sequence,
                    matched,
                    total_edits,
                )
            )
    return max(completed, default=None, key=lambda item: item[0])


def aligned_repeat_pool(probabilities, indices, records):
    probability_by_index = {int(index): probability.copy() for index, probability in zip(indices, probabilities)}
    round_values = rounds(indices, records)
    aligned = aligned_repeat_groups(
        round_values,
        probability_by_index,
        maximum_gap=120,
        minimum_similarity=0.8,
        minimum_coverage=0.9,
        maximum_length_difference=1,
        gap_penalty=0.1,
    )
    for group, matches in aligned:
        mappings = [{first: second for first, second, _ in matched} for matched in matches]
        for reference_position, reference_index in enumerate(group[0]["rows"]):
            matched_indices = [reference_index]
            for item, mapping in zip(group[1:], mappings):
                if reference_position in mapping:
                    matched_indices.append(item["rows"][mapping[reference_position]])
            if len(matched_indices) < 2:
                continue
            pooled = power_pool(np.stack([probability_by_index[index] for index in matched_indices]), 1.5)
            for index in matched_indices:
                probability_by_index[index] = pooled
    return probability_by_index, round_values, [group for group, _ in aligned]


def prepare_fold(
    probabilities,
    indices,
    labels,
    records,
    sources,
    excluded_users,
    aligned_repeats=False,
):
    probability_by_index, round_values, groups = (
        aligned_repeat_pool(probabilities, indices, records)
        if aligned_repeats
        else repeat_pool(probabilities, indices, records)
    )
    source_users = {user for values in sources for user in values}
    templates = {
        user: sorted({sequence for values in user_templates(records, user).values() for sequence in values})
        for user in source_users
    }
    transitions = [
        transition_matrix(records, set(excluded_users), environment=value) for value in (0, 1)
    ]
    group_by_round = {}
    for group_index, group in enumerate(groups):
        for item in group:
            group_by_round[tuple(item["rows"])] = group_index
    prepared_rounds = []
    for item in round_values:
        user = records[item["rows"][0]]["user"]
        environment = int(user > 15)
        values = np.stack([probability_by_index[index] for index in item["rows"]])
        transition = transitions[environment]
        free = assign_transition_unique(values, transition, 0.4, beam_size=100).argmax(axis=1)
        unary = np.log(np.clip(values, 1e-12, 1.0))
        prepared_rounds.append(
            {
                "rows": item["rows"],
                "user": user,
                "start": item["start"],
                "end": item["end"],
                "group": group_by_round.get(tuple(item["rows"])),
                "environment": environment,
                "values": values,
                "transition": transition,
                "free": tuple(int(value) for value in free),
                "free_score": model_score(free, unary, transition, 0.4),
            }
        )
    return {
        "probabilities": probabilities,
        "indices": indices,
        "labels": labels,
        "users": np.asarray([records[int(index)]["user"] for index in indices]),
        "sources": sources,
        "templates": templates,
        "cooccurrence": cooccurrence_matrix(records, set(excluded_users)),
        "rounds": prepared_rounds,
        "repeat_groups": len(groups),
    }


def evaluate(
    prepared,
    template_strength,
    gap_penalty,
    maximum_edits,
    allowed_users=None,
    alignment_beam_size=80,
    source_by_user=None,
    minimum_margin=0.0,
    allow_substitution=False,
    insertion_context_strength=0.0,
    insertion_context_pool="mean",
):
    probabilities = prepared["probabilities"]
    indices = prepared["indices"]
    labels = prepared["labels"]
    users = prepared["users"]
    sources = prepared["sources"]
    templates = prepared["templates"]
    predictions = {int(index): int(probability.argmax()) for index, probability in zip(indices, probabilities)}
    selected_rounds = 0
    changed = 0
    matched_tokens = 0
    edit_count = 0
    selection_by_user = defaultdict(lambda: {"selected_rounds": 0, "changed_predictions": 0, "margin": 0.0})
    for item in prepared["rounds"]:
        environment = item["environment"]
        values = item["values"]
        transition = item["transition"]
        selected = item["free"]
        best = (item["free_score"], item["free_score"], selected, 0, 0)
        if allowed_users is None or item["user"] in allowed_users:
            candidate_sources = (
                [source_by_user[item["user"]]]
                if source_by_user is not None and item["user"] in source_by_user
                else sources[environment]
            )
            for source in candidate_sources:
                for template in templates[source]:
                    insertion_prior = None
                    if insertion_context_strength:
                        context = prepared["cooccurrence"][:, np.asarray(template, dtype=np.int64)]
                        insertion_prior = (
                            context.max(axis=1)
                            if insertion_context_pool == "max"
                            else context.mean(axis=1)
                        )
                    candidate = flexible_assignment(
                        values,
                        template,
                        transition,
                        0.4,
                        template_strength,
                        gap_penalty,
                        maximum_edits,
                        beam_size=alignment_beam_size,
                        allow_substitution=allow_substitution,
                        insertion_prior=insertion_prior,
                        insertion_strength=insertion_context_strength,
                    )
                    if candidate is not None and candidate[0] > best[0]:
                        best = candidate
        if best[2] != selected and best[0] - item["free_score"] >= minimum_margin:
            selected_rounds += 1
            matched_tokens += best[3]
            edit_count += best[4]
            changed += sum(left != right for left, right in zip(selected, best[2]))
            user_values = selection_by_user[item["user"]]
            user_values["selected_rounds"] += 1
            user_values["changed_predictions"] += sum(left != right for left, right in zip(selected, best[2]))
            user_values["margin"] += float(best[0] - item["free_score"])
            selected = best[2]
        predictions.update(zip(item["rows"], selected))
    prediction = np.asarray([predictions[int(index)] for index in indices])
    per_user = {}
    for user in sorted(set(users.tolist())):
        selected = users == user
        per_user[str(user)] = {
            "rows": int(selected.sum()),
            "correct": int((prediction[selected] == labels[selected]).sum()),
            "accuracy": float((prediction[selected] == labels[selected]).mean()),
        } | selection_by_user[user]
    return {
        "correct": int((prediction == labels).sum()),
        "accuracy": float((prediction == labels).mean()),
        "predictions": prediction.tolist(),
        "selected_rounds": selected_rounds,
        "changed_predictions": changed,
        "matched_tokens": matched_tokens,
        "edit_count": edit_count,
        "rounds": len(prepared["rounds"]),
        "repeat_groups": prepared["repeat_groups"],
        "per_user": per_user,
    }


def main():
    args = parse_args()
    records = get_records()
    folds = [selected_probabilities(fold) for fold in (0, 1)]
    fold0 = prepare_fold(*folds[0], records, PAIR_SOURCES[0], FOLDS[0])
    candidates = []
    for maximum_edits in (1, 2):
        for gap_penalty in (0.1, 0.3):
            for template_strength in (0.35, 0.5, 0.65, 0.8):
                candidates.append(
                    {
                        "maximum_edits": maximum_edits,
                        "gap_penalty": gap_penalty,
                        "template_strength": float(template_strength),
                        "fold0": evaluate(
                            fold0,
                            float(template_strength),
                            gap_penalty,
                            maximum_edits,
                        ),
                    }
                )
                print(json.dumps(candidates[-1]), flush=True)
    selected = max(
        candidates,
        key=lambda item: (
            item["fold0"]["correct"],
            -item["fold0"]["changed_predictions"],
            -item["maximum_edits"],
            -item["template_strength"],
        ),
    )
    fold1 = prepare_fold(*folds[1], records, PAIR_SOURCES[1], FOLDS[1])
    selected["fold1"] = evaluate(
        fold1,
        selected["template_strength"],
        selected["gap_penalty"],
        selected["maximum_edits"],
    )
    correct = selected["fold0"]["correct"] + selected["fold1"]["correct"]
    rows = len(folds[0][2]) + len(folds[1][2])
    report = {
        "protocol": (
            "Edit-aware paired-template settings selected on fold0; fold1 is blind. "
            "Transitions exclude validation users. Paired labels emulate labeled-source to unseen-target transfer."
        ),
        "selected": selected,
        "combined": {"correct": correct, "rows": rows, "accuracy": correct / rows},
        "top_fold0": sorted(
            candidates,
            key=lambda item: (item["fold0"]["correct"], -item["fold0"]["changed_predictions"]),
            reverse=True,
        )[:20],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "top_fold0"}, indent=2))


if __name__ == "__main__":
    main()

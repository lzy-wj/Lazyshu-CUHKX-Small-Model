#!/usr/bin/env python3
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.blend_aware_v12.audit_existing import align, apply_auxiliary, load, temperature_scale
from scripts.evaluate_repeat_decoder import (
    pool_repeat_probabilities,
    record_interval,
    repeat_groups,
    rounds,
)
from src.data import FOLDS, get_records


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/scene_context_v13/outputs/sequence_decoder.json",
    )
    return parser.parse_args()


def selected_probabilities(fold):
    reference = load(ROOT / f"outputs/study_r2plus1d/lightweight_base_int8/base_fold{fold}.npz")
    old_r2 = align(
        reference,
        load(ROOT / f"experiments/blend_aware_v12/outputs/predictions/v12_lowrank04_old_f{fold}.npz"),
    )
    mixup_name = "v12_mixup_f0_selected_raw_int8.npz" if fold == 0 else "v12_mixup_f1_selected_raw_int8.npz"
    mixup_r2 = align(
        reference,
        load(ROOT / "experiments/blend_aware_v12/outputs/predictions" / mixup_name),
    )
    auxiliary = align(
        reference,
        load(ROOT / f"outputs/study_x3dl_soup/selected_cropblend50_fold{fold}.npz"),
    )
    r2 = 0.875 * mixup_r2 + 0.125 * old_r2
    main = 0.4 * reference["probabilities"] + 0.6 * temperature_scale(r2, 2.0)
    probabilities = apply_auxiliary(main, auxiliary)
    return probabilities, reference["indices"], reference["labels"]


def training_sequences(records, excluded_users, environment):
    sequences = []
    users = sorted(
        {
            record["user"]
            for record in records
            if record["split"] == "train"
            and record["user"] not in excluded_users
            and int(record["user"] > 15) == environment
        }
    )
    for user in users:
        indices = [
            record["index"]
            for record in records
            if record["split"] == "train" and record["user"] == user
        ]
        grouped = defaultdict(list)
        for item in rounds(indices, records):
            labels = tuple(records[index]["label"] for index in item["rows"])
            prefixes = Counter(
                "-".join(records[index]["trial"].split("-")[:2]) for index in item["rows"]
            )
            grouped[prefixes.most_common(1)[0][0]].append(labels)
        for values in grouped.values():
            sequences.append(Counter(values).most_common(1)[0][0])
    return sequences


class NgramPrior:
    def __init__(self, sequences, order, concentration):
        self.order = order
        self.concentration = concentration
        self.counts = [defaultdict(lambda: np.zeros(41, dtype=np.float64)) for _ in range(order)]
        for sequence in sequences:
            tokens = (40,) * (order - 1) + sequence
            for position in range(order - 1, len(tokens)):
                label = tokens[position]
                for context_size in range(order):
                    context = tuple(tokens[position - context_size:position]) if context_size else ()
                    self.counts[context_size][context][label] += 1
            end_tokens = tokens + (40,)
            position = len(end_tokens) - 1
            for context_size in range(order):
                context = tuple(end_tokens[position - context_size:position]) if context_size else ()
                self.counts[context_size][context][40] += 1

    def log_probabilities(self, history):
        base = self.counts[0][()].copy() + 0.5
        probability = base / base.sum()
        maximum = min(self.order - 1, len(history))
        for context_size in range(1, maximum + 1):
            context = tuple(history[-context_size:])
            counts = self.counts[context_size].get(context)
            if counts is None or counts.sum() == 0:
                continue
            probability = (counts + self.concentration * probability) / (counts.sum() + self.concentration)
        return np.log(np.clip(probability, 1e-12, 1.0))


def sequence_assignment(probabilities, prior, strength, beam_size):
    unary = np.log(np.clip(probabilities, 1e-12, 1.0))
    beams = [(0.0, (), 0)]
    for position in range(len(probabilities)):
        candidates = []
        for score, sequence, used in beams:
            values = unary[position] + strength * prior.log_probabilities(sequence)[:40]
            for label in np.argsort(values)[::-1]:
                label = int(label)
                if not (used >> label) & 1:
                    candidates.append(
                        (score + values[label], sequence + (label,), used | (1 << label))
                    )
        candidates.sort(key=lambda item: item[0], reverse=True)
        beams = candidates[:beam_size]
    scored = []
    for score, sequence, used in beams:
        score += strength * prior.log_probabilities(sequence)[40]
        scored.append((score, sequence, used))
    return np.asarray(max(scored, key=lambda item: item[0])[1], dtype=np.int64)


def repeat_pool(probabilities, indices, records):
    probability_by_index = {
        int(index): probability.copy() for index, probability in zip(indices, probabilities)
    }
    round_values = rounds(indices, records)
    groups = repeat_groups(round_values, probability_by_index, 120, 0.7)
    for group in groups:
        if len({len(item["rows"]) for item in group}) != 1:
            continue
        pooled = pool_repeat_probabilities(
            np.stack(
                [[probability_by_index[index] for index in item["rows"]] for item in group]
            ),
            power=1.5,
        )
        for item in group:
            for position, index in enumerate(item["rows"]):
                probability_by_index[index] = pooled[position]
    return probability_by_index, round_values, groups


def evaluate(probabilities, indices, labels, records, priors, strength, beam_size):
    probability_by_index, round_values, groups = repeat_pool(probabilities, indices, records)
    predictions = {}
    for item in round_values:
        environment = int(records[item["rows"][0]]["user"] > 15)
        values = np.stack([probability_by_index[index] for index in item["rows"]])
        assigned = sequence_assignment(values, priors[environment], strength, beam_size)
        predictions.update(zip(item["rows"], assigned.tolist()))
    prediction = np.asarray(
        [predictions.get(int(index), probability.argmax()) for index, probability in zip(indices, probabilities)]
    )
    return {
        "accuracy": float((prediction == labels).mean()),
        "correct": int((prediction == labels).sum()),
        "rounds": len(round_values),
        "repeat_groups": len(groups),
    }


def main():
    args = parse_args()
    records = get_records()
    folds = [selected_probabilities(fold) for fold in (0, 1)]
    candidates = []
    for order in (2, 3, 4):
        for concentration in (2.0, 8.0):
            priors = [
                NgramPrior(
                    training_sequences(records, set(FOLDS[0]), environment),
                    order,
                    concentration,
                )
                for environment in (0, 1)
            ]
            for strength in np.arange(0.1, 0.81, 0.1):
                result = evaluate(*folds[0], records, priors, float(strength), 100)
                candidates.append(
                    {
                        "order": order,
                        "concentration": concentration,
                        "strength": float(strength),
                        "fold0": result,
                    }
                )
    selected = max(
        candidates,
        key=lambda item: (
            item["fold0"]["correct"],
            -item["strength"],
            -item["order"],
            -item["concentration"],
        ),
    )
    priors = [
        NgramPrior(
            training_sequences(records, set(FOLDS[1]), environment),
            selected["order"],
            selected["concentration"],
        )
        for environment in (0, 1)
    ]
    selected["fold1"] = evaluate(
        *folds[1], records, priors, selected["strength"], 100
    )
    report = {
        "protocol": "N-gram settings selected on fold0; fold1 evaluated once with fold-local training labels only.",
        "selected": selected,
        "combined": {
            "correct": selected["fold0"]["correct"] + selected["fold1"]["correct"],
            "rows": len(folds[0][2]) + len(folds[1][2]),
        },
        "top_fold0": sorted(
            candidates,
            key=lambda item: item["fold0"]["correct"],
            reverse=True,
        )[:25],
    }
    report["combined"]["accuracy"] = report["combined"]["correct"] / report["combined"]["rows"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "top_fold0"}, indent=2))


if __name__ == "__main__":
    main()

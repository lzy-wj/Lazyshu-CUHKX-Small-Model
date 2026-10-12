#!/usr/bin/env python3
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.blend_aware_v12.research_aligned_decoder import load, align, temperature_scale
from scripts.evaluate_repeat_decoder import decode, rounds, transition_matrix
from src.data import FOLDS, get_indices, get_records


def final_probabilities(fold):
    base = load(ROOT / f"outputs/study_r2plus1d/lightweight_base_int8/base_fold{fold}.npz")
    auxiliary = align(base, load(ROOT / f"outputs/study_x3dl_soup/selected_cropblend50_fold{fold}.npz"))
    mixup = align(
        base,
        load(ROOT / "experiments/blend_aware_v12/outputs/predictions"
             / f"v12_mixup_f{fold}_selected_raw_int8.npz"),
    )
    target = align(
        base,
        load(ROOT / "experiments/blend_aware_v12/outputs/predictions"
             / f"v12_lowrank04_old_f{fold}.npz"),
    )
    candidate = 0.875 * mixup + 0.125 * target
    probabilities = 0.4 * base["probabilities"] + 0.6 * temperature_scale(candidate, 2.0)
    gated = probabilities.max(axis=1) < 0.15
    probabilities[gated] = 0.9 * probabilities[gated] + 0.1 * auxiliary[gated]
    return probabilities, base["indices"], base["labels"]


def position_prior(records, excluded_users, environment, smoothing=0.5):
    indices = [
        record["index"] for record in records
        if record["split"] == "train" and record["user"] not in excluded_users
        and (record["user"] > 15) == bool(environment)
    ]
    background = np.full(40, smoothing, dtype=np.float64)
    counts = defaultdict(lambda: None)
    support = defaultdict(int)
    for item in rounds(indices, records):
        length = len(item["rows"])
        if counts[length] is None:
            counts[length] = np.full((length, 40), smoothing, dtype=np.float64)
        labels = [records[index]["label"] for index in item["rows"]]
        background[labels] += 1
        counts[length][np.arange(length), labels] += 1
        support[length] += 1
    background /= background.sum()
    priors = {}
    for length, values in counts.items():
        conditional = values / values.sum(axis=1, keepdims=True)
        priors[length] = np.clip(np.log(conditional / background), -4, 4)
    return priors, dict(support)


def main():
    records = get_records()
    probabilities = [final_probabilities(fold) for fold in (0, 1)]
    transitions = [
        [transition_matrix(records, set(FOLDS[fold]), environment=value) for value in (0, 1)]
        for fold in (0, 1)
    ]
    priors = []
    support = []
    for fold in (0, 1):
        fold_priors = []
        fold_support = []
        for environment in (0, 1):
            current, current_support = position_prior(records, set(FOLDS[fold]), environment)
            fold_priors.append(current)
            fold_support.append(current_support)
        priors.append(fold_priors)
        support.append(fold_support)
    candidates = []
    for strength in np.linspace(0, 1.0, 21):
        result = decode(
            probabilities[0][0], probabilities[0][1], records,
            maximum_gap=120, minimum_similarity=0.7, enforce_unique=True,
            transition_strength=0.4, environment_transitions=transitions[0],
            repeat_pool_power=1.5, position_priors=priors[0], position_strength=float(strength),
        )
        candidates.append({
            "position_strength": float(strength),
            "fold0_correct": round(result["accuracy"] * len(probabilities[0][2])),
            "fold0_accuracy": result["accuracy"],
        })
    selected = max(candidates, key=lambda item: (item["fold0_correct"], -item["position_strength"]))
    blind = decode(
        probabilities[1][0], probabilities[1][1], records,
        maximum_gap=120, minimum_similarity=0.7, enforce_unique=True,
        transition_strength=0.4, environment_transitions=transitions[1],
        repeat_pool_power=1.5, position_priors=priors[1],
        position_strength=selected["position_strength"],
    )
    report = {
        "protocol": "Position-prior strength selected on fold0 only; fold1 blind",
        "selected": selected,
        "fold1": {
            "correct": round(blind["accuracy"] * len(probabilities[1][2])),
            "accuracy": blind["accuracy"],
        },
        "combined": {
            "correct": selected["fold0_correct"] + round(blind["accuracy"] * len(probabilities[1][2])),
            "rows": len(probabilities[0][2]) + len(probabilities[1][2]),
            "lowrank_reference_correct": 1103,
        },
        "training_round_support": support,
        "candidates": candidates,
    }
    report["combined"]["accuracy"] = report["combined"]["correct"] / report["combined"]["rows"]
    report["combined"]["delta_correct"] = report["combined"]["correct"] - 1103
    output = ROOT / "experiments/blend_aware_v12/outputs/position_decoder_blind.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "candidates"}, indent=2))


if __name__ == "__main__":
    main()

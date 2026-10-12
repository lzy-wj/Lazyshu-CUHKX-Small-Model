#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.research_flexible_pair_decoder import evaluate, prepare_fold
from experiments.scene_context_v13.research_paired_script_decoder import PAIR_SOURCES
from experiments.scene_context_v13.research_pairaware_base_blend import PAIRED_USERS
from src.data import FOLDS, get_records


WEIGHTS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def domain(user, paired_users):
    return f"{'paired' if user in paired_users else 'unmatched'}_env{int(user > 15)}"


def decode(probabilities, archive, fold, records):
    prepared = prepare_fold(
        probabilities,
        archive["indices"],
        archive["labels"],
        records,
        PAIR_SOURCES[fold],
        FOLDS[fold],
    )
    return evaluate(
        prepared,
        template_strength=1.8,
        gap_penalty=0.8,
        maximum_edits=5,
        allowed_users=PAIRED_USERS[fold],
        alignment_beam_size=40,
    )


def load_fold(fold):
    base = np.load(ROOT / f"experiments/scene_context_v13/outputs/v23_parameter_soup_warm60_f{fold}.npz")
    adapted = np.load(
        ROOT / f"experiments/v24_pseudo_validation/outputs/v23_warm60_pseudo_adapt_f{fold}.npz"
    )
    if not np.array_equal(base["indices"], adapted["indices"]):
        raise ValueError(f"Fold {fold} archives are not aligned")
    return base, adapted


def apply_weights(base, adapted, records, fold, settings):
    probabilities = np.empty_like(adapted["probabilities"])
    for position, index in enumerate(adapted["indices"]):
        user = records[int(index)]["user"]
        weight = settings[domain(user, PAIRED_USERS[fold])]
        probabilities[position] = (
            (1 - weight) * base["probabilities"][position]
            + weight * adapted["probabilities"][position]
        )
    return probabilities


def main():
    records = get_records()
    folds = [load_fold(fold) for fold in (0, 1)]
    fold0_users = sorted({records[int(index)]["user"] for index in folds[0][0]["indices"]})
    candidates = []
    for weight in WEIGHTS:
        base, adapted = folds[0]
        probabilities = (1 - weight) * base["probabilities"] + weight * adapted["probabilities"]
        result = decode(probabilities, adapted, 0, records)
        candidates.append({"adapted_weight": weight, "fold0": result})
        print(json.dumps({"adapted_weight": weight, "correct": result["correct"]}), flush=True)
    settings = {}
    selection = {}
    for user in fold0_users:
        name = domain(user, PAIRED_USERS[0])
        selected = max(
            candidates,
            key=lambda item: (
                item["fold0"]["per_user"][str(user)]["correct"],
                item["adapted_weight"],
            ),
        )
        settings[name] = selected["adapted_weight"]
        selection[name] = {
            "fold0_user": user,
            "adapted_weight": selected["adapted_weight"],
            "correct": selected["fold0"]["per_user"][str(user)]["correct"],
        }
    base0, adapted0 = folds[0]
    fold0_probabilities = apply_weights(base0, adapted0, records, 0, settings)
    fold0 = decode(fold0_probabilities, adapted0, 0, records)
    base1, adapted1 = folds[1]
    fold1_probabilities = apply_weights(base1, adapted1, records, 1, settings)
    fold1 = decode(fold1_probabilities, adapted1, 1, records)
    correct = fold0["correct"] + fold1["correct"]
    output_root = ROOT / "experiments/v32_domain_adaptation_gate/outputs"
    output_root.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_root / "v32_domain_gate_f0.npz",
        probabilities=fold0_probabilities,
        indices=adapted0["indices"],
        labels=adapted0["labels"],
    )
    np.savez(
        output_root / "v32_domain_gate_f1.npz",
        probabilities=fold1_probabilities,
        indices=adapted1["indices"],
        labels=adapted1["labels"],
    )
    report = {
        "protocol": "Domain weights selected independently on Fold 0 and transferred once to Fold 1.",
        "settings": settings,
        "selection": selection,
        "fold0": fold0,
        "fold1": fold1,
        "combined": {"correct": correct, "rows": 1357, "accuracy": correct / 1357},
        "reference": {"correct": 1168, "rows": 1357},
        "promote": correct > 1168,
        "uniform_candidates": candidates,
    }
    (output_root / "search.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "uniform_candidates"}, indent=2))


if __name__ == "__main__":
    main()

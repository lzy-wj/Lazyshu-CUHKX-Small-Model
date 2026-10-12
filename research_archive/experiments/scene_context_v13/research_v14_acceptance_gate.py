#!/usr/bin/env python3
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.blend_aware_v12.audit_existing import align, load
from experiments.scene_context_v13.evaluate_candidate_domain_pairaware import apply_setting
from experiments.scene_context_v13.research_domain_base_blend import group_name
from experiments.scene_context_v13.research_flexible_pair_decoder import flexible_assignment, prepare_fold
from experiments.scene_context_v13.research_paired_script_decoder import PAIR_SOURCES
from experiments.scene_context_v13.research_pairaware_base_blend import PAIRED_USERS
from experiments.scene_context_v13.research_sequence_decoder import selected_probabilities
from src.data import FOLDS, get_records


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-pattern",
        default=str(
            ROOT
            / "experiments/scene_context_v13/outputs/predictions/"
            "v14_fixedsoup_pseudo_c0p6_e2_composite_f{fold}.npz"
        ),
    )
    parser.add_argument(
        "--settings",
        type=Path,
        default=ROOT
        / "experiments/scene_context_v13/outputs/"
        "shared6_mixup_fixedsoup_uniform_l3int6_l4b0_domain_pairaware.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/scene_context_v13/outputs/v14_acceptance_gate.json",
    )
    return parser.parse_args()


def decode(prepared, allowed_users):
    probabilities = prepared["probabilities"]
    indices = prepared["indices"]
    predictions = {
        int(index): int(probability.argmax())
        for index, probability in zip(indices, probabilities)
    }
    for item in prepared["rounds"]:
        selected = item["free"]
        best = (item["free_score"], item["free_score"], selected, 0, 0)
        if item["user"] in allowed_users:
            for source in prepared["sources"][item["environment"]]:
                for template in prepared["templates"][source]:
                    candidate = flexible_assignment(
                        item["values"],
                        template,
                        item["transition"],
                        transition_strength=0.4,
                        template_strength=1.8,
                        gap_penalty=0.8,
                        maximum_edits=5,
                        beam_size=40,
                    )
                    if candidate is not None and candidate[0] > best[0]:
                        best = candidate
        if best[2] != selected:
            selected = best[2]
        predictions.update(zip(item["rows"], selected))
    return np.asarray([predictions[int(index)] for index in indices], dtype=np.int64)


def fold_values(fold, candidate_path, settings, records):
    base, indices, labels = selected_probabilities(fold)
    reference = {"probabilities": base, "indices": indices, "labels": labels}
    candidate = align(reference, load(candidate_path))
    users = np.asarray([records[int(index)]["user"] for index in indices])
    blended = base.copy()
    for user in FOLDS[fold]:
        rows = users == user
        blended, _ = apply_setting(blended, candidate, settings[group_name(user, fold)], rows)
    prepared = prepare_fold(
        blended,
        indices,
        labels,
        records,
        PAIR_SOURCES[fold],
        FOLDS[fold],
    )
    new = decode(prepared, PAIRED_USERS[fold])
    old_archive = np.load(
        ROOT
        / f"experiments/scene_context_v13/outputs/predictions/v13_pairaware_validation_f{fold}.npz"
    )
    if not np.array_equal(indices, old_archive["indices"]):
        raise ValueError("V13 pseudo-label archive is not aligned")
    old = old_archive["pseudo_labels"]
    positions = np.arange(len(indices))
    features = {
        "candidate_agrees": candidate.argmax(axis=1) == new,
        "blend_agrees": blended.argmax(axis=1) == new,
        "base_agrees": base.argmax(axis=1) == new,
        "candidate_new_minus_old": candidate[positions, new] - candidate[positions, old],
        "blend_new_minus_old": blended[positions, new] - blended[positions, old],
        "base_confidence": base.max(axis=1),
        "candidate_confidence": candidate.max(axis=1),
    }
    return {
        "indices": indices,
        "labels": labels,
        "users": users,
        "old": old,
        "new": new,
        "features": features,
    }


def rules():
    output = [
        {"name": "all"},
        {"name": "candidate_agrees"},
        {"name": "blend_agrees"},
        {"name": "candidate_and_blend_agree"},
    ]
    for threshold in (-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2):
        output.append({"name": "candidate_margin", "threshold": threshold})
        output.append({"name": "blend_margin", "threshold": threshold})
    for threshold in (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5):
        output.append({"name": "base_low_confidence", "threshold": threshold})
    return output


def accepted(values, rule):
    features = values["features"]
    if rule["name"] == "all":
        return np.ones(len(values["old"]), dtype=bool)
    if rule["name"] == "candidate_agrees":
        return features["candidate_agrees"]
    if rule["name"] == "blend_agrees":
        return features["blend_agrees"]
    if rule["name"] == "candidate_and_blend_agree":
        return features["candidate_agrees"] & features["blend_agrees"]
    if rule["name"] == "candidate_margin":
        return features["candidate_new_minus_old"] >= rule["threshold"]
    if rule["name"] == "blend_margin":
        return features["blend_new_minus_old"] >= rule["threshold"]
    if rule["name"] == "base_low_confidence":
        return features["base_confidence"] <= rule["threshold"]
    raise ValueError(rule)


def evaluate_rule(values, rule):
    changed = values["old"] != values["new"]
    use_new = changed & accepted(values, rule)
    prediction = values["old"].copy()
    prediction[use_new] = values["new"][use_new]
    per_user = {}
    for user in sorted(set(values["users"].tolist())):
        rows = values["users"] == user
        per_user[str(user)] = {
            "correct": int((prediction[rows] == values["labels"][rows]).sum()),
            "changed": int(use_new[rows].sum()),
        }
    return {
        "correct": int((prediction == values["labels"]).sum()),
        "changed": int(use_new.sum()),
        "per_user": per_user,
    }


def changed_diagnostics(values):
    changed = values["old"] != values["new"]
    output = defaultdict(lambda: {"rows": 0, "old_correct": 0, "new_correct": 0})
    for position in np.flatnonzero(changed):
        group = str(values["users"][position])
        output[group]["rows"] += 1
        output[group]["old_correct"] += int(values["old"][position] == values["labels"][position])
        output[group]["new_correct"] += int(values["new"][position] == values["labels"][position])
    return dict(output)


def main():
    args = parse_args()
    records = get_records()
    settings = json.loads(args.settings.read_text())["domain_settings"]
    folds = [
        fold_values(fold, Path(args.candidate_pattern.format(fold=fold)), settings, records)
        for fold in (0, 1)
    ]
    candidates = []
    for rule in rules():
        candidates.append(rule | {"fold0": evaluate_rule(folds[0], rule)})
    selected = max(
        candidates,
        key=lambda item: (item["fold0"]["correct"], -item["fold0"]["changed"]),
    )
    selected["fold1"] = evaluate_rule(folds[1], selected)
    for candidate in candidates:
        candidate["fold1_audit"] = evaluate_rule(folds[1], candidate)
    correct = selected["fold0"]["correct"] + selected["fold1"]["correct"]
    rows = sum(len(values["labels"]) for values in folds)
    report = {
        "protocol": "A one-dimensional acceptance rule is selected on fold0 and transferred once to fold1 blind.",
        "selected": selected,
        "combined": {"correct": correct, "rows": rows, "accuracy": correct / rows},
        "changed_diagnostics": [changed_diagnostics(values) for values in folds],
        "top_fold0": sorted(
            candidates,
            key=lambda item: (item["fold0"]["correct"], -item["fold0"]["changed"]),
            reverse=True,
        )[:10],
        "top_combined_audit": sorted(
            candidates,
            key=lambda item: (
                item["fold0"]["correct"] + item["fold1_audit"]["correct"],
                -item["fold0"]["changed"] - item["fold1_audit"]["changed"],
            ),
            reverse=True,
        )[:10],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

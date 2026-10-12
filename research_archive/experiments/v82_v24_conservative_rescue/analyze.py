#!/usr/bin/env python3
import itertools
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.build_pair_aware_submission import SOURCE_BY_DATE
from experiments.scene_context_v13.research_flexible_pair_decoder import flexible_assignment, prepare_fold
from experiments.scene_context_v13.research_paired_script_decoder import PAIR_SOURCES
from experiments.scene_context_v13.research_pairaware_base_blend import PAIRED_USERS
from experiments.scene_context_v13.research_repeat_recovery_decoder import prepool
from scripts.evaluate_repeat_decoder import record_interval
from src.data import FOLDS, get_records


DOMAIN_WEIGHTS = {0: 0.0, 1: 76 / 405, 2: 199 / 405, 3: 130 / 405}
V59_SETTINGS = {"maximum_gap": 150, "minimum_similarity": 0.7, "power": 1.0}


def load(path):
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def align(reference, archive, key="probabilities"):
    positions = {int(index): position for position, index in enumerate(archive["indices"])}
    return archive[key][[positions[int(index)] for index in reference]]


def decode(probabilities, indices, labels, fold, records):
    prepared = prepare_fold(
        probabilities,
        indices,
        labels,
        records,
        PAIR_SOURCES[fold],
        FOLDS[fold],
    )
    prediction = {int(index): int(values.argmax()) for index, values in zip(indices, probabilities)}
    for item in prepared["rounds"]:
        selected = item["free"]
        best = (item["free_score"], item["free_score"], selected, 0, 0)
        if item["user"] in PAIRED_USERS[fold]:
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
            selected = best[2]
        prediction.update(zip(item["rows"], selected))
    return np.asarray([prediction[int(index)] for index in indices], dtype=np.int64)


def validation_domains(indices, fold, records):
    return np.asarray(
        [
            int(records[int(index)]["user"] > 15)
            + 2 * int(records[int(index)]["user"] in PAIRED_USERS[fold])
            for index in indices
        ],
        dtype=np.int64,
    )


def fold_experts(fold, records):
    anchor = load(
        ROOT / f"experiments/v24_pseudo_validation/outputs/v23_warm60_pseudo_adapt_f{fold}.npz"
    )
    indices = anchor["indices"].astype(np.int64)
    labels = anchor["labels"].astype(np.int64)
    base = load(ROOT / f"experiments/scene_context_v13/outputs/v23_parameter_soup_warm60_f{fold}.npz")
    v13 = load(
        ROOT / f"experiments/scene_context_v13/outputs/predictions/v13_pairaware_validation_f{fold}.npz"
    )
    v26 = load(
        ROOT / f"experiments/v26_system_distillation/outputs/predictions/v26_system_distill_f{fold}.npz"
    )
    v29 = load(ROOT / f"experiments/v29_crop_system_fusion/outputs/v29_fused_f{fold}.npz")
    v52 = load(ROOT / f"experiments/v52_domain_logit_residual/outputs/gated_f{fold}.npz")
    v57 = load(ROOT / f"experiments/v57_round_transformer_distillation/outputs/predictions_f{fold}.npz")
    v71 = load(ROOT / f"experiments/v71_gated_film_distillation/outputs/student_f{fold}.npz")
    anchor_probabilities = anchor["probabilities"].astype(np.float32)
    v57_probabilities = align(indices, v57).astype(np.float32)
    v71_probabilities = align(indices, v71).astype(np.float32)
    v59_probabilities, _ = prepool(v57_probabilities, indices, records, **V59_SETTINGS)
    v71_pooled, _ = prepool(v71_probabilities, indices, records, **V59_SETTINGS)
    predictions = {
        "V13": align(indices, v13, "pseudo_labels").astype(np.int64),
        "V23-base": decode(align(indices, base), indices, labels, fold, records),
        "V24": decode(anchor_probabilities, indices, labels, fold, records),
        "V26": decode(align(indices, v26), indices, labels, fold, records),
        "V29": decode(align(indices, v29), indices, labels, fold, records),
        "V52": decode(align(indices, v52), indices, labels, fold, records),
        "V57": v57_probabilities.argmax(axis=1),
        "V59": decode(v59_probabilities, indices, labels, fold, records),
        "V71": decode(v71_pooled, indices, labels, fold, records),
    }
    expected = {
        0: {"V13": 550, "V23-base": 540, "V24": 550, "V26": 550, "V29": 550, "V52": 568, "V57": 567, "V59": 569, "V71": 570},
        1: {"V13": 611, "V23-base": 613, "V24": 618, "V26": 620, "V29": 627, "V52": 625, "V57": 628, "V59": 628, "V71": 631},
    }
    actual = {name: int((prediction == labels).sum()) for name, prediction in predictions.items()}
    if actual != expected[fold]:
        raise ValueError(f"Fold {fold} reproduction failed: {actual}")
    sorted_probabilities = np.sort(anchor_probabilities, axis=1)
    return {
        "indices": indices,
        "labels": labels,
        "domains": validation_domains(indices, fold, records),
        "anchor_confidence": sorted_probabilities[:, -1],
        "anchor_margin": sorted_probabilities[:, -1] - sorted_probabilities[:, -2],
        "predictions": predictions,
    }


def weighted_accuracy(prediction, fold):
    values = []
    for domain, weight in DOMAIN_WEIGHTS.items():
        selected = fold["domains"] == domain
        if selected.any() and weight:
            values.append(weight * float((prediction[selected] == fold["labels"][selected]).mean()))
    return sum(values)


def consensus(fold, names):
    votes = np.stack([fold["predictions"][name] for name in names], axis=1)
    alternate = np.empty(len(votes), dtype=np.int64)
    count = np.empty(len(votes), dtype=np.int64)
    for position, row in enumerate(votes):
        ranked = Counter(row.tolist()).most_common()
        anchor = fold["predictions"]["V24"][position]
        choice = next(((label, total) for label, total in ranked if label != anchor), (anchor, 0))
        alternate[position], count[position] = choice
    return alternate, count


def apply_rule(fold, names, settings):
    anchor = fold["predictions"]["V24"]
    adaptation_preserved = anchor != fold["predictions"]["V23-base"]
    alternate, count = consensus(fold, names)
    selected = (
        (count >= settings["minimum_votes"])
        & (fold["anchor_confidence"] <= settings["maximum_confidence"])
        & (fold["anchor_margin"] <= settings["maximum_margin"])
        & np.isin(fold["domains"], settings["domains"])
        & (alternate != anchor)
        & ~adaptation_preserved
    )
    prediction = anchor.copy()
    prediction[selected] = alternate[selected]
    return prediction, selected, alternate, count


def metrics(prediction, selected, fold):
    anchor = fold["predictions"]["V24"]
    labels = fold["labels"]
    return {
        "correct": int((prediction == labels).sum()),
        "anchor_correct": int((anchor == labels).sum()),
        "delta_correct": int((prediction == labels).sum() - (anchor == labels).sum()),
        "weighted_accuracy": weighted_accuracy(prediction, fold),
        "anchor_weighted_accuracy": weighted_accuracy(anchor, fold),
        "weighted_delta": weighted_accuracy(prediction, fold) - weighted_accuracy(anchor, fold),
        "changed": int(selected.sum()),
        "fixed": int(((anchor != labels) & (prediction == labels)).sum()),
        "broken": int(((anchor == labels) & (prediction != labels)).sum()),
        "changed_by_domain": {
            str(domain): int((selected & (fold["domains"] == domain)).sum()) for domain in (1, 2, 3)
        },
    }


def test_domains(records, paths):
    by_path = {}
    for record in records:
        if record["split"] != "test":
            continue
        date = datetime.fromtimestamp(record_interval(record)[0]).strftime("%Y-%m-%d")
        by_path[record["id"]] = int(date >= "2025-06-09") + 2 * int(SOURCE_BY_DATE[date] is not None)
    return np.asarray([by_path[path] for path in paths], dtype=np.int64)


def test_experts(records):
    submissions = {
        "V13": "outputs/submissions/v13_pair_aware.csv",
        "V23-base": "experiments/v24_full_refit/outputs/submissions/v24_full_warm60_int8_pair_aware.csv",
        "V24": "experiments/v24_full_refit/outputs/submissions/v24_full_warm60_pseudo_pair_aware.csv",
        "V26": "experiments/v24_full_refit/outputs/submissions/v26_system_distill_pair_aware.csv",
        "V29": "experiments/v24_full_refit/outputs/submissions/v29_crop_system_fusion_pair_aware.csv",
        "V52": "experiments/v24_full_refit/outputs/submissions/v52_domain_logit_residual_pair_aware.csv",
        "V57": "experiments/v24_full_refit/outputs/submissions/v57_round_transformer_distilled.csv",
        "V59": "experiments/v24_full_refit/outputs/submissions/v59_teacher_sequence_hybrid.csv",
        "V71": "experiments/v24_full_refit/outputs/submissions/v71_gated_film_sequence_hybrid.csv",
    }
    tables = {name: pd.read_csv(ROOT / path) for name, path in submissions.items()}
    reference = tables["V24"]
    for name, table in tables.items():
        if not table["path"].equals(reference["path"]):
            raise ValueError(f"Test paths do not align for {name}")
    anchor = load(ROOT / "experiments/v24_full_refit/outputs/predictions/v24_full_warm60_pseudo_adapt.npz")
    records_by_id = {record["id"]: record["index"] for record in records if record["split"] == "test"}
    indices = np.asarray([records_by_id[path] for path in reference["path"]], dtype=np.int64)
    probabilities = align(indices, anchor).astype(np.float32)
    sorted_probabilities = np.sort(probabilities, axis=1)
    return {
        "paths": reference["path"].to_numpy(),
        "domains": test_domains(records, reference["path"]),
        "anchor_confidence": sorted_probabilities[:, -1],
        "anchor_margin": sorted_probabilities[:, -1] - sorted_probabilities[:, -2],
        "predictions": {name: table["prediction"].to_numpy(np.int64) for name, table in tables.items()},
    }


def main():
    records = get_records()
    folds = [fold_experts(fold, records) for fold in (0, 1)]
    expert_names = ("V13", "V23-base", "V26", "V29", "V52", "V57", "V59", "V71")
    oracle = []
    for fold in folds:
        anchor = fold["predictions"]["V24"]
        stack = np.stack([fold["predictions"][name] for name in expert_names], axis=1)
        any_correct = (stack == fold["labels"][:, None]).any(axis=1)
        oracle_prediction = anchor.copy()
        oracle_prediction[(anchor != fold["labels"]) & any_correct] = fold["labels"][(anchor != fold["labels"]) & any_correct]
        oracle.append(
            {
                "anchor_correct": int((anchor == fold["labels"]).sum()),
                "oracle_correct": int((oracle_prediction == fold["labels"]).sum()),
                "recoverable_errors": int(((anchor != fold["labels"]) & any_correct).sum()),
            }
        )
    candidates = []
    domain_sets = [(1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
    for minimum_votes, confidence, margin, domains in itertools.product(
        range(3, len(expert_names) + 1),
        (0.35, 0.5, 0.65, 0.8),
        (0.05, 0.1, 0.2, 0.35, 0.6),
        domain_sets,
    ):
        settings = {
            "minimum_votes": minimum_votes,
            "maximum_confidence": confidence,
            "maximum_margin": margin,
            "domains": domains,
        }
        prediction, selected, _, _ = apply_rule(folds[0], expert_names, settings)
        result = metrics(prediction, selected, folds[0])
        candidates.append({"settings": settings, "fold0": result})
    selected = max(
        candidates,
        key=lambda item: (
            item["fold0"]["weighted_delta"],
            item["fold0"]["delta_correct"],
            -item["fold0"]["changed"],
            item["settings"]["minimum_votes"],
            -item["settings"]["maximum_confidence"],
            -item["settings"]["maximum_margin"],
        ),
    )
    fold1_prediction, fold1_selected, _, _ = apply_rule(folds[1], expert_names, selected["settings"])
    selected["fold1"] = metrics(fold1_prediction, fold1_selected, folds[1])
    test = test_experts(records)
    test_prediction, test_selected, test_alternate, test_votes = apply_rule(test, expert_names, selected["settings"])
    changes = []
    for position in np.flatnonzero(test_selected):
        changes.append(
            {
                "position": int(position),
                "path": str(test["paths"][position]),
                "domain": int(test["domains"][position]),
                "v24": int(test["predictions"]["V24"][position]),
                "candidate": int(test_prediction[position]),
                "votes": int(test_votes[position]),
                "anchor_confidence": float(test["anchor_confidence"][position]),
                "anchor_margin": float(test["anchor_margin"][position]),
            }
        )
    report = {
        "protocol": "Consensus and thresholds are selected on test-domain-weighted Fold 0, then transferred once to Fold 1 and unlabeled test.",
        "adaptation_guard": "Never override a row where V24 differs from V23-base; V24 adaptation gained 8/201 public rows over that base.",
        "experts": expert_names,
        "oracle": oracle,
        "selected": selected,
        "combined_delta_correct": selected["fold0"]["delta_correct"] + selected["fold1"]["delta_correct"],
        "combined_changed": selected["fold0"]["changed"] + selected["fold1"]["changed"],
        "test_changes": changes,
        "test_changed": len(changes),
        "top_fold0": sorted(
            candidates,
            key=lambda item: (item["fold0"]["weighted_delta"], item["fold0"]["delta_correct"], -item["fold0"]["changed"]),
            reverse=True,
        )[:50],
    }
    output = HERE / "outputs/analysis.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "top_fold0"}, indent=2))


if __name__ == "__main__":
    main()

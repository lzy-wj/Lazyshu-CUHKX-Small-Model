#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.research_repeat_recovery_decoder import run
from experiments.v109_crossfit_round_transformer.train import v104_probabilities
from experiments.v114_paired_person_fusion.evaluate import fused_probabilities
from experiments.v125_domain_complement.evaluate import scores
from experiments.v70_film_margin_distillation.validate import SETTINGS as V71_SETTINGS
from experiments.v84_residual_invariant_skeleton.evaluate import PAIR_SETTINGS, domains
from src.data import get_records


FAMILIES = {
    "desk": tuple(range(17, 28)),
    "interaction": (
        0, 1, 2, 4, 6, 7, 8, 9, 10, 11, 14, 17, 18, 19, 20, 21, 22, 23,
        24, 26, 27, 37, 38, 39,
    ),
}
WEIGHTS = (0.1, 0.25, 0.5, 0.75, 1.0)
ROUTES = ("unmatched", "paired", "both")


def load(path):
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def align(reference, candidate):
    positions = {int(index): position for position, index in enumerate(candidate["indices"])}
    order = np.asarray([positions[int(index)] for index in reference["indices"]])
    if not np.array_equal(candidate["labels"][order], reference["labels"]):
        raise ValueError("Validation labels do not align")
    return candidate["probabilities"][order].astype(np.float64)


def conditional_blend(base, expert, labels, weight):
    labels = np.asarray(labels, dtype=np.int64)
    output = base.copy()
    base_mass = base[:, labels].sum(axis=1, keepdims=True)
    expert_conditional = expert[:, labels]
    expert_conditional /= expert_conditional.sum(axis=1, keepdims=True).clip(1e-12)
    output[:, labels] = (
        (1 - weight) * base[:, labels] + weight * base_mass * expert_conditional
    )
    output /= output.sum(axis=1, keepdims=True)
    return output


def components(fold, records):
    reference, _, v104, _ = v104_probabilities(fold, records)
    v71 = load(
        ROOT / f"experiments/v71_gated_film_distillation/outputs/student_f{fold}.npz"
    )
    if not np.array_equal(v71["indices"], reference["indices"]):
        raise ValueError("V71 indices do not align")
    v71_probabilities = v71["probabilities"].astype(np.float64)
    v114_probabilities, _ = fused_probabilities(fold, reference, v104, records, 0.6)
    domain = domains(reference["indices"], fold, records)
    return reference, v71_probabilities, v114_probabilities, domain


def decode(fold, reference, v71, v114, domain, records):
    v71_result = run(
        fold,
        (v71, reference["indices"], reference["labels"]),
        records,
        V71_SETTINGS,
    )
    v114_result = run(
        fold,
        (v114, reference["indices"], reference["labels"]),
        records,
        PAIR_SETTINGS,
    )
    return np.where(
        domain >= 2,
        np.asarray(v114_result["predictions"], dtype=np.int64),
        np.asarray(v71_result["predictions"], dtype=np.int64),
    )


def family_accuracy(probabilities, labels, family):
    selected = np.isin(labels, family)
    family = np.asarray(family, dtype=np.int64)
    prediction = family[probabilities[selected][:, family].argmax(axis=1)]
    return {
        "correct": int((prediction == labels[selected]).sum()),
        "rows": int(selected.sum()),
        "accuracy": float((prediction == labels[selected]).mean()),
    }


def candidate(fold, family_name, route, weight, values, records):
    reference, v71, v114, domain, expert = values
    labels = FAMILIES[family_name]
    if route in ("unmatched", "both"):
        v71 = conditional_blend(v71, expert, labels, weight)
    if route in ("paired", "both"):
        v114 = conditional_blend(v114, expert, labels, weight)
    prediction = decode(fold, reference, v71, v114, domain, records)
    return scores(prediction, reference, fold, records)


def main():
    records = get_records()
    folds = {}
    family_diagnostics = {}
    for fold in (0, 1):
        reference, v71, v114, domain = components(fold, records)
        routed = np.where((domain >= 2)[:, None], v114, v71)
        folds[fold] = {}
        for family_name, family in FAMILIES.items():
            expert = align(
                reference,
                load(
                    HERE
                    / f"outputs/checkpoints/v135_{family_name}_f{fold}/validation.npz"
                ),
            )
            folds[fold][family_name] = (reference, v71, v114, domain, expert)
            family_diagnostics[f"{family_name}_f{fold}"] = {
                "incumbent_conditional": family_accuracy(
                    routed, reference["labels"], family
                ),
                "expert_conditional": family_accuracy(
                    expert, reference["labels"], family
                ),
            }

    reference0, v710, v1140, domain0, _ = folds[0]["desk"]
    control0 = scores(
        decode(0, reference0, v710, v1140, domain0, records),
        reference0,
        0,
        records,
    )
    grid = [{"family": "control", "route": "none", "weight": 0.0, "fold0": control0}]
    for family_name in FAMILIES:
        for route in ROUTES:
            for weight in WEIGHTS:
                grid.append(
                    {
                        "family": family_name,
                        "route": route,
                        "weight": weight,
                        "fold0": candidate(
                            0, family_name, route, weight, folds[0][family_name], records
                        ),
                    }
                )
    selected = max(
        grid,
        key=lambda item: (
            item["fold0"]["correct"],
            item["fold0"]["test_domain_weighted_accuracy"],
            -item["weight"],
        ),
    )
    if selected["family"] == "control":
        reference1, v711, v1141, domain1, _ = folds[1]["desk"]
        fold1 = scores(
            decode(1, reference1, v711, v1141, domain1, records),
            reference1,
            1,
            records,
        )
    else:
        fold1 = candidate(
            1,
            selected["family"],
            selected["route"],
            selected["weight"],
            folds[1][selected["family"]],
            records,
        )
    correct = selected["fold0"]["correct"] + fold1["correct"]
    rows = selected["fold0"]["rows"] + fold1["rows"]
    report = {
        "protocol": (
            "Family, causal route, and coarse mass-redistribution weight are selected "
            "on Fold0; the exact setting is transferred once to Fold1."
        ),
        "families": {key: list(value) for key, value in FAMILIES.items()},
        "weights": WEIGHTS,
        "routes": ROUTES,
        "family_diagnostics": family_diagnostics,
        "grid": grid,
        "selected": selected | {"fold1": fold1},
        "summary": {"correct": correct, "rows": rows, "accuracy": correct / rows},
        "v127": {"correct": 1205, "rows": 1357},
        "promote": correct > 1205,
    }
    output = HERE / "outputs/evaluation.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "family_diagnostics": family_diagnostics,
                "selected": report["selected"],
                "summary": report["summary"],
                "promote": report["promote"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

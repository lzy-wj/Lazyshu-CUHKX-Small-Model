#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.v84_residual_invariant_skeleton.evaluate import align, load, pair_metrics, raw_metrics
from src.data import get_records


DESK_LABELS = np.arange(17, 28)


def temperature_scale(probabilities, temperature):
    logits = np.log(np.clip(probabilities, 1e-12, 1)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    output = np.exp(logits)
    return output / output.sum(axis=1, keepdims=True)


def conditional_blend(base, expert, labels, weight):
    output = base.copy()
    base_mass = base[:, labels].sum(axis=1, keepdims=True)
    base_conditional = base[:, labels] / np.clip(base_mass, 1e-12, None)
    expert_values = expert[:, labels]
    expert_conditional = expert_values / np.clip(expert_values.sum(axis=1, keepdims=True), 1e-12, None)
    output[:, labels] = base_mass * (
        (1 - weight) * base_conditional + weight * expert_conditional
    )
    return output


def fold_values(fold):
    reference = load(
        ROOT / f"experiments/scene_context_v13/outputs/v23_parameter_soup_warm60_f{fold}.npz"
    )
    lowres = align(
        reference,
        load(
            ROOT
            / f"experiments/v24_pseudo_validation/outputs/v23_warm60_pseudo_adapt_f{fold}.npz"
        ),
    ).astype(np.float64)
    highres = align(
        reference,
        load(
            ROOT
            / "experiments/v88_highres_pseudo_validation/outputs"
            / f"v23_warm60_pseudo_adapt_highres160_f{fold}.npz"
        ),
    ).astype(np.float64)
    pose = align(
        reference,
        load(HERE / f"outputs/v105_posec3d_heatmap_f{fold}/validation.npz"),
    ).astype(np.float64)
    return reference, 0.5 * (lowres + highres), pose


def candidate(base, pose, setting):
    expert = temperature_scale(pose, setting["temperature"])
    if setting["mode"] == "global":
        return (1 - setting["weight"]) * base + setting["weight"] * expert
    return conditional_blend(base, expert, DESK_LABELS, setting["weight"])


def metrics(probabilities, reference, fold, records):
    return {
        "raw": raw_metrics(
            probabilities,
            reference["indices"],
            reference["labels"],
            fold,
            records,
        ),
        "pair": pair_metrics(
            probabilities,
            reference["indices"],
            reference["labels"],
            fold,
            records,
        ),
    }


def main():
    records = get_records()
    reference0, base0, pose0 = fold_values(0)
    settings = []
    for temperature in (1.0, 2.0, 4.0):
        for weight in (0.02, 0.05, 0.1, 0.15):
            settings.append({"mode": "global", "temperature": temperature, "weight": weight})
        for weight in (0.1, 0.25, 0.5):
            settings.append({"mode": "desk_conditional", "temperature": temperature, "weight": weight})
    candidates = []
    for setting in settings:
        result = metrics(candidate(base0, pose0, setting), reference0, 0, records)
        candidates.append(setting | {"fold0": result})
        print(
            json.dumps(
                setting
                | {
                    "pair": result["pair"]["correct"],
                    "weighted": result["pair"]["test_domain_weighted_accuracy"],
                }
            ),
            flush=True,
        )
    baseline0 = metrics(base0, reference0, 0, records)
    selected = max(
        candidates,
        key=lambda item: (
            item["fold0"]["pair"]["correct"],
            item["fold0"]["pair"]["test_domain_weighted_accuracy"],
            -item["weight"],
            -item["temperature"],
        ),
    )
    reference1, base1, pose1 = fold_values(1)
    selected["fold1"] = metrics(candidate(base1, pose1, selected), reference1, 1, records)
    baseline1 = metrics(base1, reference1, 1, records)
    baseline_correct = baseline0["pair"]["correct"] + baseline1["pair"]["correct"]
    candidate_correct = selected["fold0"]["pair"]["correct"] + selected["fold1"]["pair"]["correct"]
    report = {
        "protocol": "Blend mode, temperature, and weight selected on Fold0; Fold1 evaluated once blind.",
        "standalone": [
            json.loads((HERE / f"outputs/v105_posec3d_heatmap_f{fold}/completion.json").read_text())["metrics"]
            for fold in (0, 1)
        ],
        "baseline": {"fold0": baseline0, "fold1": baseline1, "pair_correct": baseline_correct},
        "selected": selected,
        "combined": {
            "pair_correct": candidate_correct,
            "rows": len(reference0["labels"]) + len(reference1["labels"]),
            "delta_vs_v89": candidate_correct - baseline_correct,
            "mean_fold_test_domain_weighted_accuracy": float(
                np.mean(
                    [
                        selected["fold0"]["pair"]["test_domain_weighted_accuracy"],
                        selected["fold1"]["pair"]["test_domain_weighted_accuracy"],
                    ]
                )
            ),
        },
        "candidates": candidates,
    }
    output = HERE / "outputs/evaluation.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "candidates"}, indent=2))


if __name__ == "__main__":
    main()

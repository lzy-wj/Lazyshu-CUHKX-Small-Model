#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.v125_domain_complement.evaluate import scores
from experiments.v135_end_to_end_label_experts.evaluate import align, components, decode, load
from experiments.v164_evidence_gated_radar.evaluate import accept
from experiments.v165_balanced_cycle_radar.evaluate import cycle_mask
from experiments.v166_repeated_transition_consensus.evaluate import fold_inputs, repeated_transition_mask
from experiments.v84_residual_invariant_skeleton.evaluate import pair_metrics, raw_metrics
from src.data import get_records


VARIANTS = ("relative", "dual")
ROUTES = ("unmatched", "paired", "both")
WEIGHTS = (0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30)
TEMPERATURES = (0.75, 1.0, 1.5, 2.0)
V127_CORRECT = (569, 636)
V166_CORRECT = (574, 652)


def candidate_path(variant, fold):
    return ROOT / "outputs/checkpoints" / f"v181_{variant}_f{fold}" / "validation.npz"


def temperature_scale(probabilities, temperature):
    logits = np.log(probabilities.clip(1e-12)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    output = np.exp(logits)
    return output / output.sum(axis=1, keepdims=True)


def fixed_v166(fold, prediction, records):
    reference, evidence, incumbent, challenger = fold_inputs(fold, records)
    groups = [records[int(index)]["user"] for index in reference["indices"]]
    selected = cycle_mask(reference["indices"], incumbent, challenger, records)
    selected |= accept(incumbent, challenger, evidence)
    selected |= repeated_transition_mask(groups, incumbent, challenger, 2)
    output = prediction.copy()
    output[selected] = challenger[selected]
    return output


def evaluate_candidate(fold, variant, route, weight, temperature, records):
    reference, v71, v114, domain = components(fold, records)
    expert = align(reference, load(candidate_path(variant, fold)))
    expert = temperature_scale(expert, temperature)
    if route in ("unmatched", "both"):
        v71 = (1 - weight) * v71 + weight * expert
    if route in ("paired", "both"):
        v114 = (1 - weight) * v114 + weight * expert
    prediction = decode(fold, reference, v71, v114, domain, records)
    labels = reference["labels"].astype(np.int64)
    incumbent = np.load(
        ROOT / "experiments/v127_supervision_routed_system/outputs" / f"v127_validation_f{fold}.npz"
    )["pseudo_labels"].astype(np.int64)
    v166_prediction = fixed_v166(fold, prediction, records)
    v166_incumbent = fixed_v166(fold, incumbent, records)
    changed = prediction != incumbent
    v166_changed = v166_prediction != v166_incumbent
    return {
        "v127": scores(prediction, reference, fold, records),
        "v166": scores(v166_prediction, reference, fold, records),
        "v127_changes": int(changed.sum()),
        "v127_gains": int((changed & (prediction == labels) & (incumbent != labels)).sum()),
        "v127_losses": int((changed & (prediction != labels) & (incumbent == labels)).sum()),
        "v166_changes": int(v166_changed.sum()),
        "v166_gains": int((v166_changed & (v166_prediction == labels) & (v166_incumbent != labels)).sum()),
        "v166_losses": int((v166_changed & (v166_prediction != labels) & (v166_incumbent == labels)).sum()),
    }


def standalone(fold, variant, records):
    reference, _, _, _ = components(fold, records)
    candidate = align(reference, load(candidate_path(variant, fold)))
    return {
        "raw": raw_metrics(candidate, reference["indices"], reference["labels"], fold, records),
        "pair": pair_metrics(candidate, reference["indices"], reference["labels"], fold, records),
    }


def main():
    records = get_records()
    variants = {}
    for variant in VARIANTS:
        fold0_grid = []
        for route in ROUTES:
            for weight in WEIGHTS:
                for temperature in TEMPERATURES:
                    fold0_grid.append({
                        "route": route,
                        "weight": weight,
                        "temperature": temperature,
                        "result": evaluate_candidate(0, variant, route, weight, temperature, records),
                    })
        selected = max(
            fold0_grid,
            key=lambda item: (
                item["result"]["v166"]["correct"],
                item["result"]["v127"]["correct"],
                -item["result"]["v166_losses"],
                -item["result"]["v166_changes"],
                -item["weight"],
            ),
        )
        fold1 = evaluate_candidate(
            1, variant, selected["route"], selected["weight"], selected["temperature"], records
        )
        summary = {
            "v127_correct": selected["result"]["v127"]["correct"] + fold1["v127"]["correct"],
            "v166_correct": selected["result"]["v166"]["correct"] + fold1["v166"]["correct"],
            "rows": 1357,
        }
        variants[variant] = {
            "standalone": {"fold0": standalone(0, variant, records), "fold1": standalone(1, variant, records)},
            "selected": selected | {"fold1": fold1},
            "summary": summary,
            "promote": (
                selected["result"]["v166"]["correct"] >= V166_CORRECT[0]
                and fold1["v166"]["correct"] >= V166_CORRECT[1]
                and summary["v166_correct"] > sum(V166_CORRECT)
            ),
        }
    report = {
        "protocol": "Route, weight, and temperature selected on Fold0 and transferred once to Fold1.",
        "baselines": {"v127": list(V127_CORRECT), "v166": list(V166_CORRECT)},
        "variants": variants,
    }
    (HERE / "outputs").mkdir(exist_ok=True)
    (HERE / "outputs/evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        name: {"selected": values["selected"], "summary": values["summary"], "promote": values["promote"]}
        for name, values in variants.items()
    }, indent=2))


if __name__ == "__main__":
    main()

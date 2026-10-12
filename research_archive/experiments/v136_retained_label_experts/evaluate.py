#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.v125_domain_complement.evaluate import scores
from experiments.v135_end_to_end_label_experts.evaluate import (
    align,
    components,
    conditional_blend,
    decode,
    family_accuracy,
    load,
)
from src.data import get_records


FAMILIES = {
    "documents": (17, 18, 21, 22),
    "devices": (19, 20, 23, 24, 25, 26, 27, 37, 38, 39),
}
VARIANTS = {
    "documents": ("documents",),
    "devices": ("devices",),
    "combined": ("documents", "devices"),
}
WEIGHTS = (0.1, 0.25, 0.5, 0.75)
ROUTES = ("unmatched", "paired", "both")


def apply_experts(base, experts, names, weight):
    output = base
    for name in names:
        output = conditional_blend(output, experts[name], FAMILIES[name], weight)
    return output


def evaluate_candidate(fold, values, names, route, weight, records):
    reference, v71, v114, domain, experts = values
    if route in ("unmatched", "both"):
        v71 = apply_experts(v71, experts, names, weight)
    if route in ("paired", "both"):
        v114 = apply_experts(v114, experts, names, weight)
    prediction = decode(fold, reference, v71, v114, domain, records)
    return scores(prediction, reference, fold, records)


def main():
    records = get_records()
    folds = {}
    diagnostics = {}
    for fold in (0, 1):
        reference, v71, v114, domain = components(fold, records)
        routed = np.where((domain >= 2)[:, None], v114, v71)
        experts = {}
        for family_name, family in FAMILIES.items():
            experts[family_name] = align(
                reference,
                load(
                    HERE
                    / f"outputs/checkpoints/v136_{family_name}_f{fold}/validation.npz"
                ),
            )
            diagnostics[f"{family_name}_f{fold}"] = {
                "incumbent_conditional": family_accuracy(
                    routed, reference["labels"], family
                ),
                "expert_conditional": family_accuracy(
                    experts[family_name], reference["labels"], family
                ),
            }
        folds[fold] = (reference, v71, v114, domain, experts)

    reference0, v710, v1140, domain0, _ = folds[0]
    control0 = scores(
        decode(0, reference0, v710, v1140, domain0, records),
        reference0,
        0,
        records,
    )
    grid = [{"variant": "control", "route": "none", "weight": 0.0, "fold0": control0}]
    for variant, names in VARIANTS.items():
        for route in ROUTES:
            for weight in WEIGHTS:
                grid.append(
                    {
                        "variant": variant,
                        "route": route,
                        "weight": weight,
                        "fold0": evaluate_candidate(
                            0, folds[0], names, route, weight, records
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
    if selected["variant"] == "control":
        reference1, v711, v1141, domain1, _ = folds[1]
        fold1 = scores(
            decode(1, reference1, v711, v1141, domain1, records),
            reference1,
            1,
            records,
        )
    else:
        fold1 = evaluate_candidate(
            1,
            folds[1],
            VARIANTS[selected["variant"]],
            selected["route"],
            selected["weight"],
            records,
        )
    correct = selected["fold0"]["correct"] + fold1["correct"]
    rows = selected["fold0"]["rows"] + fold1["rows"]
    report = {
        "protocol": (
            "Expert subset, causal route, and one shared coarse mass weight are "
            "selected on Fold0 and transferred unchanged to Fold1."
        ),
        "families": {key: list(value) for key, value in FAMILIES.items()},
        "variants": {key: list(value) for key, value in VARIANTS.items()},
        "weights": WEIGHTS,
        "routes": ROUTES,
        "diagnostics": diagnostics,
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
                "diagnostics": diagnostics,
                "selected": report["selected"],
                "summary": report["summary"],
                "promote": report["promote"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

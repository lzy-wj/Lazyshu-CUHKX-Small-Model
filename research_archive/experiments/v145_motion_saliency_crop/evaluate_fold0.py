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
from experiments.v84_residual_invariant_skeleton.evaluate import pair_metrics, raw_metrics
from src.data import get_records


FRACTIONS = (0.50, 0.65)
ROUTES = ("unmatched", "paired", "both")
WEIGHTS = (0.02, 0.05, 0.10, 0.20)


def candidate(values, motion, route, weight, records):
    reference, v71, v114, domain = values
    if route in ("unmatched", "both"):
        v71 = (1 - weight) * v71 + weight * motion
    if route in ("paired", "both"):
        v114 = (1 - weight) * v114 + weight * motion
    prediction = decode(0, reference, v71, v114, domain, records)
    return scores(prediction, reference, 0, records)


def main():
    records = get_records()
    values = components(0, records)
    reference, v71, v114, domain = values
    control = scores(decode(0, reference, v71, v114, domain, records), reference, 0, records)
    grid = [{"fraction": 0.0, "route": "control", "weight": 0.0, "fold0": control}]
    standalone = {}
    for fraction in FRACTIONS:
        archive = load(HERE / f"outputs/motion_{fraction:.2f}_f0.npz")
        motion = align(reference, archive)
        standalone[str(fraction)] = {
            "raw": raw_metrics(motion, reference["indices"], reference["labels"], 0, records),
            "pair": pair_metrics(motion, reference["indices"], reference["labels"], 0, records),
        }
        for route in ROUTES:
            for weight in WEIGHTS:
                grid.append(
                    {
                        "fraction": fraction,
                        "route": route,
                        "weight": weight,
                        "fold0": candidate(values, motion, route, weight, records),
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
    report = {
        "protocol": "Crop fraction, route, and low weight selected on Fold0 only.",
        "standalone": standalone,
        "grid": grid,
        "selected": selected,
        "advance_fold1": selected["route"] != "control",
    }
    output = HERE / "outputs/fold0_evaluation.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"standalone": standalone, "selected": selected, "advance_fold1": report["advance_fold1"]}, indent=2))


if __name__ == "__main__":
    main()


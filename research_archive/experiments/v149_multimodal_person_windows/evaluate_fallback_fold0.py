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


THRESHOLDS = (0.1, 0.2)
ROUTES = ("unmatched", "paired", "both")
WEIGHTS = (0.02, 0.05, 0.10, 0.20)


def main():
    records = get_records()
    reference, v71, v114, domain = components(0, records)
    control = scores(decode(0, reference, v71, v114, domain, records), reference, 0, records)
    grid = [{"threshold": 0.0, "route": "control", "weight": 0.0, "fold0": control}]
    standalone = {}
    for threshold in THRESHOLDS:
        values = align(reference, load(HERE / f"outputs/fallback_q{int(threshold * 10)}_f0.npz"))
        standalone[str(threshold)] = {
            "raw": raw_metrics(values, reference["indices"], reference["labels"], 0, records),
            "pair": pair_metrics(values, reference["indices"], reference["labels"], 0, records),
        }
        for route in ROUTES:
            for weight in WEIGHTS:
                candidate_v71 = (1 - weight) * v71 + weight * values if route in ("unmatched", "both") else v71
                candidate_v114 = (1 - weight) * v114 + weight * values if route in ("paired", "both") else v114
                prediction = decode(0, reference, candidate_v71, candidate_v114, domain, records)
                grid.append(
                    {
                        "threshold": threshold,
                        "route": route,
                        "weight": weight,
                        "fold0": scores(prediction, reference, 0, records),
                    }
                )
    selected = max(
        grid,
        key=lambda item: (item["fold0"]["correct"], item["fold0"]["test_domain_weighted_accuracy"], -item["weight"]),
    )
    report = {
        "protocol": "IR quality threshold, route, and low weight selected on Fold0 only.",
        "standalone": standalone,
        "grid": grid,
        "selected": selected,
        "advance": selected["route"] != "control",
    }
    output = HERE / "outputs/fallback_fold0_evaluation.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"standalone": standalone, "selected": selected, "advance": report["advance"]}, indent=2))


if __name__ == "__main__":
    main()


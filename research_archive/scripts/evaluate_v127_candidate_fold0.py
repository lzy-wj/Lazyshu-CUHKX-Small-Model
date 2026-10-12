#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.v125_domain_complement.evaluate import scores
from experiments.v135_end_to_end_label_experts.evaluate import align, components, decode, load
from experiments.v84_residual_invariant_skeleton.evaluate import pair_metrics, raw_metrics
from src.data import get_records


ROUTES = ("unmatched", "paired", "both")
WEIGHTS = (0.02, 0.05, 0.10, 0.20)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    records = get_records()
    reference, v71, v114, domain = components(0, records)
    candidate = align(reference, load(args.candidate))
    control = json.loads(
        (ROOT / "experiments/v127_supervision_routed_system/outputs/evaluation.json").read_text()
    )["folds"][0]["v127"]
    grid = [{"route": "control", "weight": 0.0, "fold0": control}]
    for route in ROUTES:
        for weight in WEIGHTS:
            candidate_v71 = (
                (1 - weight) * v71 + weight * candidate
                if route in ("unmatched", "both")
                else v71
            )
            candidate_v114 = (
                (1 - weight) * v114 + weight * candidate
                if route in ("paired", "both")
                else v114
            )
            prediction = decode(
                0, reference, candidate_v71, candidate_v114, domain, records
            )
            grid.append(
                {
                    "route": route,
                    "weight": weight,
                    "fold0": scores(prediction, reference, 0, records),
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
        "protocol": "Route and conservative blend weight selected on Fold0 only.",
        "candidate": str(args.candidate),
        "standalone": {
            "raw": raw_metrics(
                candidate, reference["indices"], reference["labels"], 0, records
            ),
            "pair": pair_metrics(
                candidate, reference["indices"], reference["labels"], 0, records
            ),
        },
        "grid": grid,
        "selected": selected,
        "advance": selected["route"] != "control",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "standalone": report["standalone"],
                "selected": selected,
                "advance": report["advance"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

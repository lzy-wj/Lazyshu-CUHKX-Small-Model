#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.v125_domain_complement.evaluate import scores
from experiments.v126_shared_mvit.evaluate import fold_inputs, routed_prediction
from experiments.v84_residual_invariant_skeleton.evaluate import align, pair_metrics, raw_metrics
from src.data import get_records


WEIGHTS = (0.02, 0.05, 0.1)
ROUTES = ("unmatched", "paired", "both")


def main():
    records = get_records()
    inputs = [fold_inputs(fold, records) for fold in (0, 1)]
    swin = [
        align(
            inputs[fold][0],
            dict(
                np.load(
                    ROOT
                    / "outputs/checkpoints"
                    / f"v133_shared_swin3d_f{fold}/validation.npz"
                )
            ),
        )
        for fold in (0, 1)
    ]
    standalone = []
    for fold in (0, 1):
        reference = inputs[fold][0]
        standalone.append(
            {
                "fold": fold,
                "raw": raw_metrics(
                    swin[fold], reference["indices"], reference["labels"], fold, records
                ),
                "pair": pair_metrics(
                    swin[fold], reference["indices"], reference["labels"], fold, records
                ),
            }
        )
    reference0, v710, v1140, prediction710, prediction1140 = inputs[0]
    control0 = np.where(
        np.asarray([records[int(index)]["user"] in {8, 18} for index in reference0["indices"]]),
        prediction1140,
        prediction710,
    )
    grid = [
        {
            "route": "control",
            "weight": 0.0,
            "fold0": scores(control0, reference0, 0, records),
        }
    ]
    for route in ROUTES:
        for weight in WEIGHTS:
            prediction = routed_prediction(
                0,
                records,
                reference0,
                v710,
                v1140,
                prediction710,
                prediction1140,
                swin[0],
                route,
                weight,
            )
            grid.append(
                {"route": route, "weight": weight, "fold0": scores(prediction, reference0, 0, records)}
            )
    selected = max(
        grid,
        key=lambda item: (
            item["fold0"]["correct"],
            item["fold0"]["test_domain_weighted_accuracy"],
            -item["weight"],
        ),
    )
    reference1, v711, v1141, prediction711, prediction1141 = inputs[1]
    control1 = np.where(
        np.asarray([records[int(index)]["user"] in {7, 17} for index in reference1["indices"]]),
        prediction1141,
        prediction711,
    )
    if selected["route"] == "control":
        prediction1 = control1
    else:
        prediction1 = routed_prediction(
            1,
            records,
            reference1,
            v711,
            v1141,
            prediction711,
            prediction1141,
            swin[1],
            selected["route"],
            selected["weight"],
        )
    selected["fold1"] = scores(prediction1, reference1, 1, records)
    controls = [scores(control0, reference0, 0, records), scores(control1, reference1, 1, records)]
    candidate = [selected["fold0"], selected["fold1"]]
    summary = {}
    for name, values in (("v127", controls), ("v133", candidate)):
        correct = sum(value["correct"] for value in values)
        rows = sum(value["rows"] for value in values)
        weighted = [value["test_domain_weighted_accuracy"] for value in values]
        summary[name] = {
            "correct": correct,
            "rows": rows,
            "accuracy": correct / rows,
            "fold_test_domain_weighted_accuracy": weighted,
            "mean_fold_test_domain_weighted_accuracy": float(np.mean(weighted)),
        }
    report = {
        "protocol": "Swin route and coarse low weight selected on Fold0, then transferred once to Fold1.",
        "weights": WEIGHTS,
        "routes": ROUTES,
        "standalone": standalone,
        "grid": grid,
        "selected": selected,
        "summary": summary,
        "promote": summary["v133"]["correct"] > summary["v127"]["correct"]
        and summary["v133"]["mean_fold_test_domain_weighted_accuracy"]
        > summary["v127"]["mean_fold_test_domain_weighted_accuracy"],
    }
    output = HERE / "outputs/evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selected": selected, "summary": summary, "promote": report["promote"]}, indent=2))


if __name__ == "__main__":
    main()

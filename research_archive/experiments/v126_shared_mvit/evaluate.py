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
from experiments.v84_residual_invariant_skeleton.evaluate import (
    PAIR_SETTINGS,
    align,
    domains,
    pair_metrics,
    raw_metrics,
)
from src.data import get_records


WEIGHTS = (0.05, 0.1, 0.2, 0.3)
CHECKPOINTS = {
    "full": "v126_mvit_full_f{fold}",
    "visual": "v126_mvit_visual_f{fold}",
    "domainadapt": "v128_mvit_domainadapt_f{fold}",
}
VARIANTS = (*CHECKPOINTS, "hybrid")
ROUTES = ("unmatched", "paired", "both")


def fold_inputs(fold, records):
    reference, _, v104, _ = v104_probabilities(fold, records)
    v71 = align(
        reference,
        dict(
            np.load(
                ROOT
                / "experiments/v71_gated_film_distillation/outputs"
                / f"student_f{fold}.npz"
            )
        ),
    )
    v114, _ = fused_probabilities(fold, reference, v104, records, weight=0.6)
    v71_prediction = np.asarray(
        run(
            fold,
            (v71, reference["indices"], reference["labels"]),
            records,
            V71_SETTINGS,
        )["predictions"],
        dtype=np.int64,
    )
    v114_prediction = np.asarray(
        run(
            fold,
            (v114, reference["indices"], reference["labels"]),
            records,
            PAIR_SETTINGS,
        )["predictions"],
        dtype=np.int64,
    )
    return reference, v71, v114, v71_prediction, v114_prediction


def decode(probabilities, reference, fold, records, settings):
    return np.asarray(
        run(
            fold,
            (probabilities, reference["indices"], reference["labels"]),
            records,
            settings,
        )["predictions"],
        dtype=np.int64,
    )


def routed_prediction(
    fold,
    records,
    reference,
    v71,
    v114,
    v71_prediction,
    v114_prediction,
    mvit,
    route,
    weight,
):
    unmatched = v71_prediction
    paired = v114_prediction
    if route in ("unmatched", "both"):
        unmatched = decode(
            (1 - weight) * v71 + weight * mvit,
            reference,
            fold,
            records,
            V71_SETTINGS,
        )
    if route in ("paired", "both"):
        paired = decode(
            (1 - weight) * v114 + weight * mvit,
            reference,
            fold,
            records,
            PAIR_SETTINGS,
        )
    domain = domains(reference["indices"], fold, records)
    return np.where(domain >= 2, paired, unmatched)


def main():
    records = get_records()
    inputs = [fold_inputs(fold, records) for fold in (0, 1)]
    mvit = {
        (variant, fold): align(
            inputs[fold][0],
            dict(
                np.load(
                    ROOT
                    / "outputs/checkpoints"
                    / CHECKPOINTS[variant].format(fold=fold)
                    / "validation.npz"
                )
            ),
        )
        for variant in CHECKPOINTS
        for fold in (0, 1)
    }
    for fold in (0, 1):
        reference = inputs[fold][0]
        domain = domains(reference["indices"], fold, records)
        hybrid = mvit[("full", fold)].copy()
        hybrid[domain >= 2] = mvit[("domainadapt", fold)][domain >= 2]
        mvit[("hybrid", fold)] = hybrid
    standalone = {}
    for variant in VARIANTS:
        standalone[variant] = []
        for fold in (0, 1):
            reference = inputs[fold][0]
            probabilities = mvit[(variant, fold)]
            standalone[variant].append(
                {
                    "fold": fold,
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
            )

    reference0, v710, v1140, prediction710, prediction1140 = inputs[0]
    domain0 = domains(reference0["indices"], 0, records)
    control0 = np.where(domain0 >= 2, prediction1140, prediction710)
    grid = [
        {
            "variant": "control",
            "route": "none",
            "weight": 0.0,
            "fold0": scores(control0, reference0, 0, records),
        }
    ]
    for variant in VARIANTS:
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
                    mvit[(variant, 0)],
                    route,
                    weight,
                )
                grid.append(
                    {
                        "variant": variant,
                        "route": route,
                        "weight": weight,
                        "fold0": scores(prediction, reference0, 0, records),
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
    reference1, v711, v1141, prediction711, prediction1141 = inputs[1]
    domain1 = domains(reference1["indices"], 1, records)
    control1 = np.where(domain1 >= 2, prediction1141, prediction711)
    if selected["route"] == "none":
        candidate1 = control1
    else:
        candidate1 = routed_prediction(
            1,
            records,
            reference1,
            v711,
            v1141,
            prediction711,
            prediction1141,
            mvit[(selected["variant"], 1)],
            selected["route"],
            selected["weight"],
        )
    selected["fold1"] = scores(candidate1, reference1, 1, records)
    controls = [
        scores(control0, reference0, 0, records),
        scores(control1, reference1, 1, records),
    ]
    candidate = [selected["fold0"], selected["fold1"]]
    summary = {}
    for name, values in (("v127", controls), ("v126_selected", candidate)):
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
        "protocol": (
            "MViT variant, causal route, and coarse blend are selected on Fold0; "
            "the frozen selection is evaluated once on Fold1. Control is eligible."
        ),
        "hybrid_definition": (
            "Conservative-backbone MViT on unmatched domains and stronger visual-domain "
            "adaptation on paired domains; selected from Fold0 and transferred unchanged."
        ),
        "variants": VARIANTS,
        "routes": ROUTES,
        "weights": WEIGHTS,
        "standalone": standalone,
        "grid": grid,
        "selected": selected,
        "summary": summary,
        "promote": summary["v126_selected"]["correct"] > summary["v127"]["correct"]
        and summary["v126_selected"]["mean_fold_test_domain_weighted_accuracy"]
        > summary["v127"]["mean_fold_test_domain_weighted_accuracy"],
    }
    output = HERE / "outputs/evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "selected": {
                    key: selected[key] for key in ("variant", "route", "weight")
                },
                "summary": summary,
                "promote": report["promote"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.v109_crossfit_round_transformer.train import v104_probabilities
from experiments.v114_paired_person_fusion.evaluate import fused_probabilities
from experiments.v125_domain_complement.evaluate import scores
from experiments.v84_residual_invariant_skeleton.evaluate import align, domains
from src.data import get_records


def rules():
    values = [{"name": "control"}]
    values.extend(
        {"name": "raw_confidence", "threshold": threshold}
        for threshold in (0.4, 0.5, 0.6, 0.7)
    )
    values.extend(
        {"name": "decoder_rank", "maximum_rank": rank} for rank in (2, 3, 5)
    )
    values.extend(
        {"name": "decoder_log_margin", "minimum_margin": margin}
        for margin in (-3.0, -2.0, -1.0, -0.5)
    )
    return values


def load_fold(fold, records):
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
    domain = domains(reference["indices"], fold, records)
    raw = v71.copy()
    raw[domain >= 2] = v114[domain >= 2]
    archive = np.load(
        ROOT
        / "experiments/v127_supervision_routed_system/outputs"
        / f"v127_validation_f{fold}.npz"
    )
    if not np.array_equal(archive["indices"], reference["indices"]):
        raise ValueError("V127 validation indices do not align")
    return reference, raw, archive["pseudo_labels"].astype(np.int64)


def apply_rule(raw, decoded, labels, rule):
    raw_prediction = raw.argmax(1)
    changed = decoded != raw_prediction
    revert = np.zeros(len(decoded), dtype=bool)
    if rule["name"] == "control":
        pass
    elif rule["name"] == "raw_confidence":
        revert = raw.max(1) >= rule["threshold"]
    elif rule["name"] == "decoder_rank":
        rank = np.argsort(np.argsort(-raw, axis=1), axis=1)[
            np.arange(len(decoded)), decoded
        ] + 1
        revert = rank > rule["maximum_rank"]
    elif rule["name"] == "decoder_log_margin":
        positions = np.arange(len(decoded))
        margin = np.log(np.clip(raw[positions, decoded], 1e-12, 1)) - np.log(
            np.clip(raw[positions, raw_prediction], 1e-12, 1)
        )
        revert = margin < rule["minimum_margin"]
    else:
        raise ValueError(rule)
    revert &= changed
    prediction = decoded.copy()
    prediction[revert] = raw_prediction[revert]
    return prediction, {
        "decoder_changes": int(changed.sum()),
        "reverted": int(revert.sum()),
        "decoded_correct_on_reverts": int((decoded[revert] == labels[revert]).sum()),
        "raw_correct_on_reverts": int((raw_prediction[revert] == labels[revert]).sum()),
    }


def main():
    records = get_records()
    folds = [load_fold(fold, records) for fold in (0, 1)]
    reference0, raw0, decoded0 = folds[0]
    grid = []
    for rule in rules():
        prediction, diagnostics = apply_rule(
            raw0, decoded0, reference0["labels"], rule
        )
        grid.append(
            rule
            | {
                "diagnostics": diagnostics,
                "fold0": scores(prediction, reference0, 0, records),
            }
        )
    selected = max(
        grid,
        key=lambda item: (
            item["fold0"]["correct"],
            item["fold0"]["test_domain_weighted_accuracy"],
            -item["diagnostics"]["reverted"],
        ),
    )
    selected_rule = {
        key: value
        for key, value in selected.items()
        if key not in ("diagnostics", "fold0")
    }
    reference1, raw1, decoded1 = folds[1]
    prediction1, diagnostics1 = apply_rule(
        raw1, decoded1, reference1["labels"], selected_rule
    )
    selected["fold1_diagnostics"] = diagnostics1
    selected["fold1"] = scores(prediction1, reference1, 1, records)
    controls = [
        scores(decoded0, reference0, 0, records),
        scores(decoded1, reference1, 1, records),
    ]
    candidate = [selected["fold0"], selected["fold1"]]
    summary = {}
    for name, values in (("v127", controls), ("v132", candidate)):
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
            "One row-wise decoder acceptance rule selected on Fold0 and transferred "
            "unchanged to Fold1; V127 control is eligible."
        ),
        "grid": grid,
        "selected": selected,
        "summary": summary,
        "promote": summary["v132"]["correct"] > summary["v127"]["correct"]
        and summary["v132"]["mean_fold_test_domain_weighted_accuracy"]
        > summary["v127"]["mean_fold_test_domain_weighted_accuracy"],
    }
    output = HERE / "outputs/evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selected": selected, "summary": summary, "promote": report["promote"]}, indent=2))


if __name__ == "__main__":
    main()

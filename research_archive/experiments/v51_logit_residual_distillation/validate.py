#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.v32_domain_adaptation_gate.search import apply_weights, decode
from experiments.v51_logit_residual_distillation.logit_residual import fit, predict, save
from src.data import get_records


RETENTION_WEIGHTS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def load_fold(fold):
    base = np.load(ROOT / f"experiments/scene_context_v13/outputs/v23_parameter_soup_warm60_f{fold}.npz")
    adapted = np.load(ROOT / f"experiments/v24_pseudo_validation/outputs/v23_warm60_pseudo_adapt_f{fold}.npz")
    teacher_path = ROOT / f"experiments/v26_system_distillation/outputs/targets/v26_system_teacher_f{fold}.npz"
    teacher = np.load(teacher_path)
    if not np.array_equal(base["indices"], adapted["indices"]) or not np.array_equal(base["indices"], teacher["indices"]):
        raise ValueError(f"Fold {fold} archives are not aligned")
    return base, adapted, teacher, teacher_path


def train_score(fold, retention_weight, records, domain_settings, output_root=None):
    base, adapted, teacher, teacher_path = load_fold(fold)
    model, history = fit(adapted["probabilities"], teacher["pseudo_labels"], retention_weight)
    calibrated = predict(model, adapted["probabilities"])
    values = {"probabilities": calibrated, "indices": adapted["indices"], "labels": adapted["labels"]}
    gated = apply_weights(base, values, records, fold, domain_settings)
    result = decode(gated, values, fold, records)
    result.update(
        calibrated_raw_correct=int((calibrated.argmax(1) == adapted["labels"]).sum()),
        gated_raw_correct=int((gated.argmax(1) == adapted["labels"]).sum()),
        teacher_correct=int((teacher["pseudo_labels"] == adapted["labels"]).sum()),
        teacher_imitation_correct=int((calibrated.argmax(1) == teacher["pseudo_labels"]).sum()),
        history=history,
    )
    if output_root is not None:
        np.savez(
            output_root / f"calibrated_f{fold}.npz",
            probabilities=calibrated,
            indices=adapted["indices"],
            labels=adapted["labels"],
        )
        np.savez(
            output_root / f"gated_f{fold}.npz",
            probabilities=gated,
            indices=adapted["indices"],
            labels=adapted["labels"],
        )
        save(model, output_root / f"calibrator_f{fold}.pt", retention_weight, teacher_path)
    return result


def main():
    records = get_records()
    domain_settings = json.loads(
        (ROOT / "experiments/v32_domain_adaptation_gate/outputs/search.json").read_text()
    )["settings"]
    candidates = []
    for retention_weight in RETENTION_WEIGHTS:
        result = train_score(0, retention_weight, records, domain_settings)
        candidates.append({"retention_weight": retention_weight, "fold0": result})
        print(
            json.dumps(
                {
                    "retention_weight": retention_weight,
                    "decoded": result["correct"],
                    "raw": result["gated_raw_correct"],
                }
            ),
            flush=True,
        )
    selected = max(
        candidates,
        key=lambda item: (
            item["fold0"]["correct"],
            item["retention_weight"],
        ),
    )
    output_root = ROOT / "experiments/v51_logit_residual_distillation/outputs"
    output_root.mkdir(parents=True, exist_ok=True)
    selected["fold0"] = train_score(
        0, selected["retention_weight"], records, domain_settings, output_root
    )
    selected["fold1"] = train_score(
        1, selected["retention_weight"], records, domain_settings, output_root
    )
    correct = selected["fold0"]["correct"] + selected["fold1"]["correct"]
    report = {
        "protocol": "Retention selected on Fold 0; method and retention frozen before one transductive Fold-1 audit.",
        "selected": selected,
        "combined": {"correct": correct, "rows": 1357, "accuracy": correct / 1357},
        "reference": {"name": "V32", "correct": 1174, "rows": 1357},
        "system_teacher": {"correct": 1195, "rows": 1357},
        "promote": selected["fold0"]["correct"] > 551 and selected["fold1"]["correct"] > 623,
        "candidates": candidates,
    }
    (output_root / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "candidates"}, indent=2))


if __name__ == "__main__":
    main()


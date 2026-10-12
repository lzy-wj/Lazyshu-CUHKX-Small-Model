#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.research_pairaware_base_blend import PAIRED_USERS
from experiments.v32_domain_adaptation_gate.search import apply_weights, decode, domain
from experiments.v52_domain_logit_residual.domain_residual import fit_domains, save_bundle
from src.data import get_records


def run_fold(fold, records, settings, output_root):
    base = np.load(ROOT / f"experiments/scene_context_v13/outputs/v23_parameter_soup_warm60_f{fold}.npz")
    adapted = np.load(ROOT / f"experiments/v24_pseudo_validation/outputs/v23_warm60_pseudo_adapt_f{fold}.npz")
    teacher_path = ROOT / f"experiments/v26_system_distillation/outputs/targets/v26_system_teacher_f{fold}.npz"
    teacher = np.load(teacher_path)
    if not np.array_equal(base["indices"], adapted["indices"]) or not np.array_equal(base["indices"], teacher["indices"]):
        raise ValueError(f"Fold {fold} archives are not aligned")
    groups = np.asarray(
        [domain(records[int(index)]["user"], PAIRED_USERS[fold]) for index in adapted["indices"]]
    )
    calibrated, models, group_reports = fit_domains(
        adapted["probabilities"],
        base["probabilities"],
        teacher["pseudo_labels"],
        groups,
    )
    values = {"probabilities": calibrated, "indices": adapted["indices"], "labels": adapted["labels"]}
    gated = apply_weights(base, values, records, fold, settings)
    result = decode(gated, values, fold, records)
    result.update(
        calibrated_raw_correct=int((calibrated.argmax(1) == adapted["labels"]).sum()),
        gated_raw_correct=int((gated.argmax(1) == adapted["labels"]).sum()),
        teacher_correct=int((teacher["pseudo_labels"] == adapted["labels"]).sum()),
        teacher_imitation_correct=int((calibrated.argmax(1) == teacher["pseudo_labels"]).sum()),
        domains=group_reports,
    )
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
    save_bundle(models, output_root / f"calibrators_f{fold}.pt", teacher_path)
    return result


def main():
    records = get_records()
    settings = json.loads(
        (ROOT / "experiments/v32_domain_adaptation_gate/outputs/search.json").read_text()
    )["settings"]
    output_root = ROOT / "experiments/v52_domain_logit_residual/outputs"
    output_root.mkdir(parents=True, exist_ok=True)
    fold0 = run_fold(0, records, settings, output_root)
    fold1 = run_fold(1, records, settings, output_root)
    correct = fold0["correct"] + fold1["correct"]
    report = {
        "protocol": "V51 retention and V32 domains frozen; each fold fits target-domain residuals from pseudo labels only.",
        "fold0": fold0,
        "fold1": fold1,
        "combined": {"correct": correct, "rows": 1357, "accuracy": correct / 1357},
        "references": {
            "V32": {"correct": 1174, "rows": 1357},
            "V51": {"correct": 1187, "rows": 1357},
            "system_teacher": {"correct": 1195, "rows": 1357},
        },
        "promote": fold0["correct"] > 563 and fold1["correct"] >= 624,
    }
    (output_root / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()


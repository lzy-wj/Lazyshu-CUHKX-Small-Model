#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.research_pairaware_base_blend import PAIRED_USERS
from experiments.v32_domain_adaptation_gate.search import decode
from experiments.v57_round_transformer_distillation.sequence_residual import fit, predict, save_bundle
from src.data import get_records


def domain_values(indices, records, fold):
    return np.asarray(
        [int(records[int(index)]["user"] > 15) + 2 * int(records[int(index)]["user"] in PAIRED_USERS[fold])
         for index in indices],
        dtype=np.int64,
    )


def run_fold(fold, records, output_root):
    base = np.load(ROOT / f"experiments/scene_context_v13/outputs/v23_parameter_soup_warm60_f{fold}.npz")
    adapted = np.load(ROOT / f"experiments/v24_pseudo_validation/outputs/v23_warm60_pseudo_adapt_f{fold}.npz")
    anchor = np.load(ROOT / f"experiments/v52_domain_logit_residual/outputs/gated_f{fold}.npz")
    teacher_path = ROOT / f"experiments/v26_system_distillation/outputs/targets/v26_system_teacher_f{fold}.npz"
    teacher = np.load(teacher_path)
    for archive in (adapted, anchor, teacher):
        if not np.array_equal(base["indices"], archive["indices"]):
            raise ValueError(f"Fold {fold} archives are not aligned")
    domains = domain_values(anchor["indices"], records, fold)
    model, _, history = fit(
        anchor["probabilities"],
        adapted["probabilities"],
        base["probabilities"],
        teacher["pseudo_labels"],
        anchor["indices"],
        domains,
        records,
    )
    probabilities, sequence_report = predict(
        model,
        anchor["probabilities"],
        adapted["probabilities"],
        base["probabilities"],
        anchor["indices"],
        domains,
        records,
    )
    values = {"probabilities": probabilities, "indices": anchor["indices"], "labels": anchor["labels"]}
    result = decode(probabilities, values, fold, records)
    result.update(
        raw_correct=int((probabilities.argmax(1) == anchor["labels"]).sum()),
        teacher_correct=int((teacher["pseudo_labels"] == anchor["labels"]).sum()),
        teacher_imitation_correct=int((probabilities.argmax(1) == teacher["pseudo_labels"]).sum()),
        changed_raw_vs_v52=int((probabilities.argmax(1) != anchor["probabilities"].argmax(1)).sum()),
        sequence=sequence_report,
        history=history,
    )
    np.savez(
        output_root / f"predictions_f{fold}.npz",
        probabilities=probabilities,
        indices=anchor["indices"],
        labels=anchor["labels"],
    )
    save_bundle(model, output_root / f"transformer_f{fold}.pt", teacher_path)
    return result


def main():
    records = get_records()
    output_root = HERE / "outputs"
    output_root.mkdir(parents=True, exist_ok=True)
    fold0 = run_fold(0, records, output_root)
    fold1 = run_fold(1, records, output_root)
    direct_correct = fold0["raw_correct"] + fold1["raw_correct"]
    post_pair_correct = fold0["correct"] + fold1["correct"]
    report = {
        "protocol": (
            "A single fixed round-Transformer recipe is fit independently to model-generated target-fold "
            "pseudo labels. Fold 1 labels are inspected only after its pseudo adaptation completes."
        ),
        "fold0": fold0,
        "fold1": fold1,
        "combined": {
            "correct": direct_correct,
            "rows": 1357,
            "accuracy": direct_correct / 1357,
            "decision": "Use direct Transformer output because the pseudo teacher is already sequence-decoded.",
        },
        "post_pair_decoder": {
            "correct": post_pair_correct,
            "rows": 1357,
            "accuracy": post_pair_correct / 1357,
        },
        "references": {"V52": {"correct": 1193}, "system_teacher": {"correct": 1195}},
        "promote": (
            fold0["raw_correct"] >= 567
            and fold1["raw_correct"] >= 628
            and direct_correct > 1193
        ),
    }
    (output_root / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.evaluate_second_stage_acceptance import final_base
from experiments.scene_context_v13.research_v14_acceptance_gate import fold_values
from src.data import get_records


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--base-pattern", required=True)
    parser.add_argument("--base-settings", type=Path, required=True)
    parser.add_argument("--base-acceptance", type=Path, required=True)
    parser.add_argument("--candidate-pattern", required=True)
    parser.add_argument("--candidate-settings", type=Path, required=True)
    parser.add_argument("--second-stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    records = get_records()
    base_settings = json.loads(args.base_settings.read_text())["domain_settings"]
    base_rule = json.loads(args.base_acceptance.read_text())["selected"]
    candidate_settings = json.loads(args.candidate_settings.read_text())["domain_settings"]
    second_rule = json.loads(args.second_stage.read_text())["selected"]

    base_values = fold_values(
        args.fold,
        Path(args.base_pattern.format(fold=args.fold)),
        base_settings,
        records,
    )
    base_prediction = final_base(base_values, base_rule)
    candidate_path = Path(args.candidate_pattern.format(fold=args.fold))
    candidate_values = fold_values(args.fold, candidate_path, candidate_settings, records)
    candidate_prediction = candidate_values["new"]
    archive = np.load(candidate_path)
    if not np.array_equal(archive["indices"], base_values["indices"]):
        raise ValueError("Candidate archive is not aligned with the base system")
    probabilities = archive["probabilities"]
    positions = np.arange(len(probabilities))
    candidate_margin = (
        probabilities[positions, candidate_prediction]
        - probabilities[positions, base_prediction]
    )
    use_candidate = candidate_prediction != base_prediction
    if second_rule["name"] == "candidate_agrees":
        use_candidate &= probabilities.argmax(axis=1) == candidate_prediction
    elif second_rule["name"] == "candidate_margin":
        use_candidate &= candidate_margin >= second_rule["threshold"]
    elif second_rule["name"] != "all":
        raise ValueError(f"Unsupported second-stage rule: {second_rule}")
    pseudo_labels = base_prediction.copy()
    pseudo_labels[use_candidate] = candidate_prediction[use_candidate]

    classes = probabilities.shape[1]
    hard_probabilities = np.full((len(pseudo_labels), classes), 1e-6, dtype=np.float32)
    hard_probabilities[np.arange(len(pseudo_labels)), pseudo_labels] = 1 - (classes - 1) * 1e-6
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        probabilities=hard_probabilities,
        pseudo_labels=pseudo_labels.astype(np.int64),
        confidence=np.ones(len(pseudo_labels), dtype=np.float32),
        paired=np.ones(len(pseudo_labels), dtype=bool),
        indices=base_values["indices"].astype(np.int64),
        labels=base_values["labels"].astype(np.int64),
    )
    report = {
        "fold": args.fold,
        "rows": len(pseudo_labels),
        "base_system_correct_audit_only": int(
            (base_prediction == base_values["labels"]).sum()
        ),
        "candidate_system_correct_audit_only": int(
            (candidate_prediction == base_values["labels"]).sum()
        ),
        "final_teacher_correct_audit_only": int(
            (pseudo_labels == base_values["labels"]).sum()
        ),
        "selected_candidate_rows": int(use_candidate.sum()),
        "second_stage_rule": second_rule,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()


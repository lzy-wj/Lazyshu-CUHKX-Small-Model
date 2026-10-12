#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.research_v14_acceptance_gate import accepted, fold_values
from src.data import get_records


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-pattern", required=True)
    parser.add_argument("--base-settings", type=Path, required=True)
    parser.add_argument("--base-acceptance", type=Path, required=True)
    parser.add_argument("--candidate-pattern", required=True)
    parser.add_argument("--candidate-settings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def final_base(values, rule):
    use_new = (values["old"] != values["new"]) & accepted(values, rule)
    prediction = values["old"].copy()
    prediction[use_new] = values["new"][use_new]
    return prediction


def rules():
    output = [{"name": "all"}, {"name": "candidate_agrees"}]
    output.extend(
        {"name": "candidate_margin", "threshold": threshold}
        for threshold in (-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3, 0.4)
    )
    return output


def evaluate(values, rule):
    use_new = values["base"] != values["candidate"]
    if rule["name"] == "candidate_agrees":
        use_new &= values["candidate_agrees"]
    elif rule["name"] == "candidate_margin":
        use_new &= values["candidate_margin"] >= rule["threshold"]
    prediction = values["base"].copy()
    prediction[use_new] = values["candidate"][use_new]
    return {
        "correct": int((prediction == values["labels"]).sum()),
        "changed": int(use_new.sum()),
    }


def main():
    args = parse_args()
    records = get_records()
    base_settings = json.loads(args.base_settings.read_text())["domain_settings"]
    base_rule = json.loads(args.base_acceptance.read_text())["selected"]
    candidate_settings = json.loads(args.candidate_settings.read_text())["domain_settings"]
    folds = []
    oracle = []
    for fold in (0, 1):
        base_values = fold_values(
            fold,
            Path(args.base_pattern.format(fold=fold)),
            base_settings,
            records,
        )
        base = final_base(base_values, base_rule)
        candidate_values = fold_values(
            fold,
            Path(args.candidate_pattern.format(fold=fold)),
            candidate_settings,
            records,
        )
        candidate = candidate_values["new"]
        archive = np.load(args.candidate_pattern.format(fold=fold))
        probabilities = archive["probabilities"]
        positions = np.arange(len(base))
        values = {
            "labels": base_values["labels"],
            "base": base,
            "candidate": candidate,
            "candidate_agrees": probabilities.argmax(axis=1) == candidate,
            "candidate_margin": probabilities[positions, candidate] - probabilities[positions, base],
        }
        folds.append(values)
        oracle.append(
            {
                "base_correct": int((base == values["labels"]).sum()),
                "candidate_correct": int((candidate == values["labels"]).sum()),
                "oracle_correct": int(
                    ((base == values["labels"]) | (candidate == values["labels"])).sum()
                ),
                "candidate_only": int(
                    ((base != values["labels"]) & (candidate == values["labels"])).sum()
                ),
                "base_only": int(
                    ((base == values["labels"]) & (candidate != values["labels"])).sum()
                ),
            }
        )

    candidates = [rule | {"fold0": evaluate(folds[0], rule)} for rule in rules()]
    selected = max(
        candidates,
        key=lambda item: (item["fold0"]["correct"], -item["fold0"]["changed"]),
    )
    selected["fold1"] = evaluate(folds[1], selected)
    rows = sum(len(values["labels"]) for values in folds)
    correct = selected["fold0"]["correct"] + selected["fold1"]["correct"]
    report = {
        "protocol": "One second-stage acceptance rule is selected on fold0 and transferred to fold1 blind.",
        "oracle": oracle,
        "selected": selected,
        "combined": {"correct": correct, "rows": rows, "accuracy": correct / rows},
        "fold0_candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "fold0_candidates"}, indent=2))


if __name__ == "__main__":
    main()

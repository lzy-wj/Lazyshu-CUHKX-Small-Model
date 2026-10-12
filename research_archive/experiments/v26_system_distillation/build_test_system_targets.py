#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v13-decoded", type=Path, required=True)
    parser.add_argument("--v14-decoded", type=Path, required=True)
    parser.add_argument("--v14-candidate", type=Path, required=True)
    parser.add_argument("--base-acceptance", type=Path, required=True)
    parser.add_argument("--v24-decoded", type=Path, required=True)
    parser.add_argument("--v24-candidate", type=Path, required=True)
    parser.add_argument("--second-stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_aligned(paths):
    archives = [np.load(path) for path in paths]
    indices = archives[0]["indices"]
    if any(not np.array_equal(indices, archive["indices"]) for archive in archives[1:]):
        raise ValueError("Teacher prediction archives are not aligned")
    return archives, indices


def margin_select(old, new, probabilities, rule):
    selected = old != new
    positions = np.arange(len(old))
    if rule["name"] == "candidate_margin":
        margin = probabilities[positions, new] - probabilities[positions, old]
        selected &= margin >= rule["threshold"]
    elif rule["name"] == "candidate_agrees":
        selected &= probabilities.argmax(axis=1) == new
    elif rule["name"] != "all":
        raise ValueError(f"Unsupported acceptance rule: {rule}")
    output = old.copy()
    output[selected] = new[selected]
    return output, selected


def main():
    args = parse_args()
    paths = (
        args.v13_decoded,
        args.v14_decoded,
        args.v14_candidate,
        args.v24_decoded,
        args.v24_candidate,
    )
    archives, indices = load_aligned(paths)
    v13, v14, v14_candidate, v24, v24_candidate = archives
    base_rule = json.loads(args.base_acceptance.read_text())["selected"]
    second_rule = json.loads(args.second_stage.read_text())["selected"]
    base, base_selected = margin_select(
        v13["probabilities"].argmax(axis=1),
        v14["probabilities"].argmax(axis=1),
        v14_candidate["probabilities"],
        base_rule,
    )
    pseudo_labels, second_selected = margin_select(
        base,
        v24["probabilities"].argmax(axis=1),
        v24_candidate["probabilities"],
        second_rule,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        pseudo_labels=pseudo_labels.astype(np.int64),
        confidence=np.ones(len(indices), dtype=np.float32),
        paired=np.ones(len(indices), dtype=bool),
        indices=indices.astype(np.int64),
    )
    report = {
        "rows": len(indices),
        "base_rule": base_rule,
        "base_selected_rows": int(base_selected.sum()),
        "second_stage_rule": second_rule,
        "second_stage_selected_rows": int(second_selected.sum()),
        "changed_vs_v13": int(
            (pseudo_labels != v13["probabilities"].argmax(axis=1)).sum()
        ),
        "changed_vs_v24_direct": int(
            (pseudo_labels != v24["probabilities"].argmax(axis=1)).sum()
        ),
        "protocol": "Frozen V14 base acceptance followed by frozen V24 second-stage acceptance; no test labels.",
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
V23_FOLD_CORRECT = {0: 487, 1: 567}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, default=HERE / "selection.json")
    parser.add_argument("--output-root", type=Path, default=HERE / "outputs")
    parser.add_argument("--destination", type=Path, default=HERE / "result_summary.json")
    return parser.parse_args()


def main():
    args = parse_args()
    selection = json.loads(args.selection.read_text())
    iterations = int(selection["selected_iterations"])
    folds = {}
    for fold in (0, 1):
        path = args.output_root / f"v24_iterative_k{iterations}_f{fold}" / "completion.json"
        if not path.exists():
            raise FileNotFoundError(path)
        folds[str(fold)] = json.loads(path.read_text())["metrics"]
    correct = sum(int(folds[str(fold)]["correct"]) for fold in (0, 1))
    rows = sum(int(folds[str(fold)]["rows"]) for fold in (0, 1))
    report = {
        "selected_iterations": iterations,
        "fold0_selection_gate_passed": selection["fold0_promotion_gate_passed"],
        "fold0": folds["0"],
        "fold1_blind": folds["1"],
        "combined": {
            "rows": rows,
            "correct": correct,
            "accuracy": correct / rows,
            "delta_correct_vs_v23": correct - sum(V23_FOLD_CORRECT.values()),
        },
        "fold1_delta_correct_vs_frozen_v23": int(folds["1"]["correct"]) - V23_FOLD_CORRECT[1],
        "promotion_gate_passed": (
            selection["fold0_promotion_gate_passed"] and correct > sum(V23_FOLD_CORRECT.values())
        ),
        "note": "Fold1 K and schedule were frozen by Fold0; Fold1 was evaluated only after epoch 10.",
    }
    args.destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

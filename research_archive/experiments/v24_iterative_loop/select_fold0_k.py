#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=HERE / "outputs")
    parser.add_argument("--destination", type=Path, default=HERE / "selection.json")
    return parser.parse_args()


def main():
    args = parse_args()
    candidates = []
    for iterations in (1, 2, 3):
        path = args.output_root / f"v24_iterative_k{iterations}_f0" / "completion.json"
        if not path.exists():
            raise FileNotFoundError(f"Fold0 K={iterations} is incomplete: {path}")
        result = json.loads(path.read_text())
        if result["fold"] != 0 or result["iterations"] != iterations:
            raise ValueError(f"Unexpected completion metadata: {path}")
        metrics = result["metrics"]
        candidates.append(
            {
                "iterations": iterations,
                "correct": int(metrics["correct"]),
                "nll": float(metrics["nll"]),
                "visual_missing_correct": int(metrics["visual_missing"]["correct"]),
                "path": str(path),
            }
        )
    selected = max(
        candidates,
        key=lambda item: (
            item["correct"],
            -item["nll"],
            item["visual_missing_correct"],
        ),
    )
    baseline = next(item for item in candidates if item["iterations"] == 1)
    promotion_gate = (
        selected["iterations"] != 1
        and selected["correct"] >= baseline["correct"] + 3
        and selected["nll"] <= baseline["nll"]
    )
    report = {
        "protocol": "Fold0 only: correct descending, NLL ascending, missing-visual correct descending.",
        "candidates": candidates,
        "selected_iterations": selected["iterations"],
        "fold0_promotion_gate_passed": promotion_gate,
        "fold1_authorization": "Run exactly selected_iterations once; do not run other K values on Fold1.",
    }
    args.destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

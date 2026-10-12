#!/usr/bin/env python3
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT_ROOT = HERE / "outputs"
BASELINE_CORRECT = 487
BASELINE_NLL = 1.1791210174560547


def main():
    candidates = []
    for iterations in (1, 2):
        path = OUTPUT_ROOT / f"v78_wide768_d4_k{iterations}_f0" / "completion.json"
        result = json.loads(path.read_text())
        metrics = result["metrics"]
        candidates.append(
            {
                "iterations": iterations,
                "correct": int(metrics["correct"]),
                "nll": float(metrics["nll"]),
                "visual_missing_correct": int(metrics["visual_missing"]["correct"]),
                "parameters": int(result["parameters"]),
                "deployment_bytes_with_yolo": int(result["deployment_bytes_with_yolo"]),
            }
        )
    selected = max(
        candidates,
        key=lambda item: (item["correct"], -item["nll"], item["visual_missing_correct"]),
    )
    k1, k2 = candidates
    report = {
        "protocol": "Equal-parameter K1/K2 Fold0 comparison; correct>NLL>missing-visual.",
        "baseline": {"correct": BASELINE_CORRECT, "nll": BASELINE_NLL, "rows": 652},
        "candidates": candidates,
        "selected_iterations": selected["iterations"],
        "capacity_gate_passed": (
            selected["correct"] >= BASELINE_CORRECT + 3
            and selected["nll"] <= BASELINE_NLL
        ),
        "loop_gate_passed": (
            k2["correct"] > k1["correct"] and k2["nll"] <= k1["nll"]
        ),
    }
    (HERE / "selection.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

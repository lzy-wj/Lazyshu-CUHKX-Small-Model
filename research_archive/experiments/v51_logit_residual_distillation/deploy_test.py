#!/usr/bin/env python3
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.v51_logit_residual_distillation.logit_residual import fit, predict, save


def main():
    output_root = ROOT / "experiments/v24_full_refit/outputs"
    adapted_path = output_root / "predictions/v24_full_warm60_pseudo_adapt.npz"
    teacher_path = output_root / "predictions/v26_system_teacher_targets.npz"
    adapted = np.load(adapted_path)
    teacher = np.load(teacher_path)
    if not np.array_equal(adapted["indices"], teacher["indices"]):
        raise ValueError("V24 predictions and V26 teacher targets are not aligned")
    validation = json.loads(
        (ROOT / "experiments/v51_logit_residual_distillation/outputs/validation.json").read_text()
    )
    retention_weight = validation["selected"]["retention_weight"]
    model, history = fit(
        adapted["probabilities"], teacher["pseudo_labels"], retention_weight
    )
    probabilities = predict(model, adapted["probabilities"])
    prediction_path = output_root / "predictions/v51_logit_residual_calibrated.npz"
    adapter_path = output_root / "final/v51_logit_residual_conv1d.pt"
    np.savez(prediction_path, probabilities=probabilities, indices=adapted["indices"])
    save(model, adapter_path, retention_weight, teacher_path)
    report = {
        "rows": len(probabilities),
        "retention_weight": retention_weight,
        "teacher_imitation_correct": int(
            (probabilities.argmax(1) == teacher["pseudo_labels"]).sum()
        ),
        "changed_vs_v24_adapted": int(
            (probabilities.argmax(1) != adapted["probabilities"].argmax(1)).sum()
        ),
        "adapter": str(adapter_path.relative_to(ROOT)),
        "adapter_bytes": adapter_path.stat().st_size,
        "adapter_sha256": hashlib.sha256(adapter_path.read_bytes()).hexdigest(),
        "history": history,
    }
    prediction_path.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()


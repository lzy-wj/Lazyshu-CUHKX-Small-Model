#!/usr/bin/env python3
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.build_pair_aware_submission import SOURCE_BY_DATE
from experiments.v52_domain_logit_residual.domain_residual import fit_domains, save_bundle
from scripts.evaluate_repeat_decoder import record_interval
from src.data import get_records


def main():
    records = get_records()
    output_root = ROOT / "experiments/v24_full_refit/outputs"
    base_path = output_root / "predictions/v24_full_warm60_int8_tta.npz"
    adapted_path = output_root / "predictions/v24_full_warm60_pseudo_adapt.npz"
    teacher_path = output_root / "predictions/v26_system_teacher_targets.npz"
    base = np.load(base_path)
    adapted = np.load(adapted_path)
    teacher = np.load(teacher_path)
    if not np.array_equal(base["indices"], adapted["indices"]) or not np.array_equal(base["indices"], teacher["indices"]):
        raise ValueError("Test archives are not aligned")
    groups = []
    for index in adapted["indices"]:
        start, _ = record_interval(records[int(index)])
        date = datetime.fromtimestamp(start).strftime("%Y-%m-%d")
        if date not in SOURCE_BY_DATE:
            raise ValueError(f"Unexpected test acquisition date: {date}")
        paired = SOURCE_BY_DATE[date] is not None
        environment = int(date >= "2025-06-09")
        groups.append(f"{'paired' if paired else 'unmatched'}_env{environment}")
    groups = np.asarray(groups)
    probabilities, models, reports = fit_domains(
        adapted["probabilities"],
        base["probabilities"],
        teacher["pseudo_labels"],
        groups,
    )
    prediction_path = output_root / "predictions/v52_domain_logit_residual_calibrated.npz"
    adapter_path = output_root / "final/v52_domain_logit_residual_conv1d.pt"
    np.savez(prediction_path, probabilities=probabilities, indices=adapted["indices"])
    save_bundle(models, adapter_path, teacher_path)
    report = {
        "rows": len(probabilities),
        "domains": reports,
        "teacher_imitation_correct": int(
            (probabilities.argmax(1) == teacher["pseudo_labels"]).sum()
        ),
        "changed_vs_v24_adapted": int(
            (probabilities.argmax(1) != adapted["probabilities"].argmax(1)).sum()
        ),
        "adapter": str(adapter_path.relative_to(ROOT)),
        "adapter_bytes": adapter_path.stat().st_size,
    }
    prediction_path.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()


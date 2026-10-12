#!/usr/bin/env python3
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.build_pair_aware_submission import SOURCE_BY_DATE
from experiments.v57_round_transformer_distillation.sequence_residual import (
    fit,
    load_bundle,
    predict,
    save_bundle,
)
from scripts.evaluate_repeat_decoder import record_interval
from src.data import get_records


def main():
    records = get_records()
    output_root = ROOT / "experiments/v24_full_refit/outputs"
    base = np.load(output_root / "predictions/v24_full_warm60_int8_tta.npz")
    adapted = np.load(output_root / "predictions/v24_full_warm60_pseudo_adapt.npz")
    anchor = np.load(output_root / "predictions/v52_domain_logit_residual_domain_gate.npz")
    teacher_path = output_root / "predictions/v26_system_teacher_targets.npz"
    teacher = np.load(teacher_path)
    for archive in (adapted, anchor, teacher):
        if not np.array_equal(base["indices"], archive["indices"]):
            raise ValueError("Test archives are not aligned")
    domains = []
    for position, index in enumerate(anchor["indices"]):
        start, _ = record_interval(records[int(index)])
        date = datetime.fromtimestamp(start).strftime("%Y-%m-%d")
        if date not in SOURCE_BY_DATE:
            raise ValueError(f"Unexpected test acquisition date: {date}")
        paired = SOURCE_BY_DATE[date] is not None
        domains.append(int(date >= "2025-06-09") + 2 * int(paired))
    domains = np.asarray(domains, dtype=np.int64)
    model, _, history = fit(
        anchor["probabilities"],
        adapted["probabilities"],
        base["probabilities"],
        teacher["pseudo_labels"],
        anchor["indices"],
        domains,
        records,
    )
    bundle_path = output_root / "final/v57_round_residual_transformer.pt"
    save_bundle(model, bundle_path, teacher_path)
    reloaded, bundle = load_bundle(bundle_path)
    probabilities, sequence_report = predict(
        reloaded,
        anchor["probabilities"],
        adapted["probabilities"],
        base["probabilities"],
        anchor["indices"],
        domains,
        records,
    )
    predictions = probabilities.argmax(axis=1)
    imitation = int((predictions == teacher["pseudo_labels"]).sum())
    if imitation != len(predictions):
        raise RuntimeError(f"Serialized V57 reproduces only {imitation}/{len(predictions)} teacher labels")
    prediction_path = output_root / "predictions/v57_round_transformer_distilled.npz"
    submission_path = output_root / "submissions/v57_round_transformer_distilled.csv"
    np.savez(prediction_path, probabilities=probabilities, indices=anchor["indices"])
    with submission_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "prediction"])
        for index, prediction in zip(anchor["indices"], predictions):
            writer.writerow([records[int(index)]["id"], int(prediction)])
    report = {
        "teacher_imitation_correct": imitation,
        "rows": len(predictions),
        "changed_raw_vs_v52": int(
            (predictions != anchor["probabilities"].argmax(1)).sum()
        ),
        "parameters": bundle["parameters"],
        "adapter_bytes": bundle_path.stat().st_size,
        "history": history,
        "sequence": sequence_report,
        "prediction": str(prediction_path.relative_to(ROOT)),
        "submission": str(submission_path.relative_to(ROOT)),
    }
    prediction_path.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

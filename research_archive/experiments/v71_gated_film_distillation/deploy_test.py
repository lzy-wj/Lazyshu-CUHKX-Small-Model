#!/usr/bin/env python3
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.blend_aware_v12.audit_existing import temperature_scale
from experiments.scene_context_v13.build_pair_aware_submission import SOURCE_BY_DATE
from experiments.scene_context_v13.research_repeat_recovery_decoder import prepool
from experiments.v70_film_margin_distillation.validate import MODALITIES
from experiments.v71_gated_film_distillation.gated_residual import (
    fit_gated,
    load_bundle,
    predict,
    save_bundle,
)
from scripts.evaluate_repeat_decoder import record_interval
from src.data import get_records
from src.scratch_protocol import modality_presence


def load(path):
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def align(reference_indices, archive):
    positions = {int(index): position for position, index in enumerate(archive["indices"])}
    order = np.asarray([positions[int(index)] for index in reference_indices])
    return archive["probabilities"][order].astype(np.float32)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    records = get_records()
    output_root = ROOT / "experiments/v24_full_refit/outputs"
    v57 = load(output_root / "predictions/v57_round_transformer_distilled.npz")
    indices = v57["indices"].astype(np.int64)
    v52 = load(output_root / "predictions/v52_domain_logit_residual_domain_gate.npz")
    adapted = load(output_root / "predictions/v24_full_warm60_pseudo_adapt.npz")
    base = load(output_root / "predictions/v24_full_warm60_int8_tta.npz")
    film_path = ROOT / "experiments/v68_film_full_refit/outputs/predictions/v68_film_full_raw.npz"
    film = load(film_path)
    domains = []
    for index in indices:
        interval = record_interval(records[int(index)])
        if interval is None:
            raise ValueError(f"Test row {index} has no acquisition timestamp")
        date = datetime.fromtimestamp(interval[0]).strftime("%Y-%m-%d")
        if date not in SOURCE_BY_DATE:
            raise ValueError(f"Unexpected test acquisition date: {date}")
        domains.append(
            int(date >= "2025-06-09") + 2 * int(SOURCE_BY_DATE[date] is not None)
        )
    domains = np.asarray(domains, dtype=np.int64)
    experts = [
        v57["probabilities"].astype(np.float32),
        align(indices, v52),
        align(indices, adapted),
        align(indices, base),
    ]
    teacher = experts[0].copy()
    active = domains == 2
    film_probabilities = align(indices, film)
    teacher[active] = (
        0.5 * teacher[active]
        + 0.5 * temperature_scale(film_probabilities[active], 3.0)
    )
    presence = modality_presence(MODALITIES)
    model, history = fit_gated(
        experts,
        teacher,
        indices,
        domains,
        presence[indices],
        records,
        seed=7142,
    )
    bundle_path = output_root / "final/v71_gated_film_transformer.pt"
    save_bundle(model, bundle_path, "V57 plus 0.5 temperature-3 V68 in domain 2")
    reloaded, bundle = load_bundle(bundle_path)
    student, sequence = predict(
        reloaded,
        experts,
        indices,
        domains,
        presence[indices],
        records,
    )
    raw_path = output_root / "predictions/v71_gated_film_distilled.npz"
    np.savez(raw_path, probabilities=student, indices=indices)
    teacher_raw_path = output_root / "predictions/v71_film_teacher.npz"
    np.savez(teacher_raw_path, probabilities=teacher, indices=indices)
    student_prepool, student_groups = prepool(student, indices, records, 150, 0.7, 1.0)
    teacher_prepool, teacher_groups = prepool(teacher, indices, records, 150, 0.7, 1.0)
    prepool_path = output_root / "predictions/v71_gated_film_repeat_recovered.npz"
    teacher_prepool_path = output_root / "predictions/v71_film_teacher_repeat_recovered.npz"
    np.savez(prepool_path, probabilities=student_prepool, indices=indices)
    np.savez(teacher_prepool_path, probabilities=teacher_prepool, indices=indices)
    raw_top1_matches = int((student.argmax(axis=1) == teacher.argmax(axis=1)).sum())
    prepool_top1_matches = int(
        (student_prepool.argmax(axis=1) == teacher_prepool.argmax(axis=1)).sum()
    )
    if raw_top1_matches != len(indices):
        raise RuntimeError(f"V71 student retains only {raw_top1_matches}/{len(indices)} teacher labels")
    report = {
        "teacher": str(film_path.relative_to(ROOT)),
        "training_only_teacher": True,
        "active_domain": 2,
        "active_rows": int(active.sum()),
        "domain_counts": {
            str(domain): int((domains == domain).sum()) for domain in np.unique(domains)
        },
        "raw_teacher_student_top1_matches": raw_top1_matches,
        "prepool_teacher_student_top1_matches": prepool_top1_matches,
        "teacher_student_mean_absolute_error": float(np.abs(teacher - student).mean()),
        "teacher_student_max_absolute_error": float(np.abs(teacher - student).max()),
        "student_changed_raw_vs_v57": int(
            (student.argmax(axis=1) != experts[0].argmax(axis=1)).sum()
        ),
        "teacher_changed_raw_vs_v57": int(
            (teacher.argmax(axis=1) != experts[0].argmax(axis=1)).sum()
        ),
        "student_repeat_groups": student_groups,
        "teacher_repeat_groups": teacher_groups,
        "parameters": bundle["parameters"],
        "bundle_bytes": bundle_path.stat().st_size,
        "bundle_sha256": sha256(bundle_path),
        "history": history,
        "sequence": sequence,
        "raw_prediction": str(raw_path.relative_to(ROOT)),
        "repeat_prediction": str(prepool_path.relative_to(ROOT)),
    }
    raw_path.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

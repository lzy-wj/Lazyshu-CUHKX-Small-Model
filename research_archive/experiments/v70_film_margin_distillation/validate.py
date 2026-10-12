#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.blend_aware_v12.audit_existing import temperature_scale
from experiments.scene_context_v13.research_pairaware_base_blend import PAIRED_USERS
from experiments.scene_context_v13.research_repeat_recovery_decoder import run
from experiments.v70_film_margin_distillation.soft_residual import (
    fit,
    load_bundle,
    predict,
    save_bundle,
)
from src.data import get_records
from src.scratch_protocol import modality_presence


SETTINGS = {
    "enabled": True,
    "maximum_gap": 150,
    "minimum_similarity": 0.7,
    "power": 1.0,
}
MODALITIES = ("scene_depth", "thermal", "scene_ir", "skeleton", "imu", "radar")


def load(path):
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def align(indices, archive):
    positions = {int(index): position for position, index in enumerate(archive["indices"])}
    order = np.asarray([positions[int(index)] for index in indices])
    return archive["probabilities"][order].astype(np.float32)


def main():
    records = get_records()
    presence = modality_presence(MODALITIES)
    results = []
    for fold in (0, 1):
        v57 = load(
            ROOT / f"experiments/v57_round_transformer_distillation/outputs/predictions_f{fold}.npz"
        )
        indices = v57["indices"].astype(np.int64)
        v52 = load(ROOT / f"experiments/v52_domain_logit_residual/outputs/gated_f{fold}.npz")
        adapted = load(
            ROOT / f"experiments/v24_pseudo_validation/outputs/v23_warm60_pseudo_adapt_f{fold}.npz"
        )
        base = load(
            ROOT / f"experiments/scene_context_v13/outputs/v23_parameter_soup_warm60_f{fold}.npz"
        )
        film = load(
            ROOT
            / f"experiments/v55_modality_film/outputs/checkpoints/"
            f"v55_modality_film_f{fold}/validation.npz"
        )
        domains = np.asarray(
            [
                int(records[int(index)]["user"] > 15)
                + 2 * int(records[int(index)]["user"] in PAIRED_USERS[fold])
                for index in indices
            ],
            dtype=np.int64,
        )
        teacher = v57["probabilities"].astype(np.float32).copy()
        selected = domains == 2
        teacher[selected] = (
            0.5 * teacher[selected]
            + 0.5 * temperature_scale(align(indices, film)[selected], 3.0)
        )
        experts = [
            v57["probabilities"].astype(np.float32),
            align(indices, v52),
            align(indices, adapted),
            align(indices, base),
        ]
        model, history = fit(
            experts,
            teacher,
            indices,
            domains,
            presence[indices],
            records,
            seed=7000 + fold,
        )
        student, sequence = predict(
            model,
            experts,
            indices,
            domains,
            presence[indices],
            records,
        )
        teacher_result = run(
            fold,
            (teacher, indices, v57["labels"]),
            records,
            SETTINGS,
        )
        student_result = run(
            fold,
            (student, indices, v57["labels"]),
            records,
            SETTINGS,
        )
        output = HERE / f"outputs/student_f{fold}.npz"
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output, probabilities=student, indices=indices, labels=v57["labels"])
        bundle = HERE / f"outputs/student_f{fold}.pt"
        save_bundle(model, bundle, "V57 plus 0.5 temperature-3 V55 in domain 2")
        reloaded, _ = load_bundle(bundle)
        reproduced, _ = predict(
            reloaded,
            experts,
            indices,
            domains,
            presence[indices],
            records,
        )
        serialization_error = np.abs(student - reproduced)
        serialization_top1_matches = int(
            (student.argmax(axis=1) == reproduced.argmax(axis=1)).sum()
        )
        if (
            float(serialization_error.mean()) > 1e-5
            or float(serialization_error.max()) > 1e-3
            or serialization_top1_matches != len(indices)
        ):
            raise RuntimeError("Serialized student does not reproduce validation predictions")
        results.append(
            {
                "fold": fold,
                "teacher": teacher_result,
                "student": student_result,
                "teacher_student_mean_absolute_error": float(np.abs(teacher - student).mean()),
                "teacher_student_max_absolute_error": float(np.abs(teacher - student).max()),
                "teacher_top1_imitation": int((teacher.argmax(1) == student.argmax(1)).sum()),
                "serialization_mean_absolute_error": float(serialization_error.mean()),
                "serialization_max_absolute_error": float(serialization_error.max()),
                "serialization_top1_matches": serialization_top1_matches,
                "rows": len(indices),
                "sequence": sequence,
                "history": history,
                "bundle_bytes": bundle.stat().st_size,
            }
        )
    teacher_correct = sum(item["teacher"]["correct"] for item in results)
    student_correct = sum(item["student"]["correct"] for item in results)
    report = {
        "protocol": (
            "V55 teacher rule fixed on Fold 0 and transferred to Fold 1; each student is fitted "
            "without labels on its target fold and evaluated only after distillation."
        ),
        "fold0": results[0],
        "fold1": results[1],
        "teacher_combined": {"correct": teacher_correct, "rows": 1357},
        "student_combined": {"correct": student_correct, "rows": 1357},
        "reference_v59": {"correct": 1197, "rows": 1357},
        "promote": (
            student_correct >= teacher_correct
            and results[0]["student"]["correct"] >= 570
            and results[1]["student"]["correct"] >= 631
        ),
    }
    output = HERE / "outputs/validation.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

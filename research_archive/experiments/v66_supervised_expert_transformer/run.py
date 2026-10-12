#!/usr/bin/env python3
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.build_pair_aware_submission import SOURCE_BY_DATE
from experiments.scene_context_v13.research_pairaware_base_blend import PAIRED_USERS
from experiments.scene_context_v13.research_repeat_recovery_decoder import run
from experiments.v66_supervised_expert_transformer.model import fit, predict, save_bundle
from scripts.evaluate_repeat_decoder import record_interval
from src.data import get_records
from src.scratch_protocol import modality_presence


RETENTION_WEIGHTS = (0.25, 0.5, 1.0, 2.0)
BLEND_WEIGHTS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
STEPS = 300
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


def aligned(reference_indices, archive, key="probabilities"):
    positions = {int(index): position for position, index in enumerate(archive["indices"])}
    order = np.asarray([positions[int(index)] for index in reference_indices])
    return archive[key][order].astype(np.float32)


def load_fold(fold, presence_values):
    anchor = load(ROOT / f"experiments/v52_domain_logit_residual/outputs/gated_f{fold}.npz")
    indices = anchor["indices"].astype(np.int64)
    adapted = load(
        ROOT / f"experiments/v24_pseudo_validation/outputs/v23_warm60_pseudo_adapt_f{fold}.npz"
    )
    base = load(ROOT / f"experiments/scene_context_v13/outputs/v23_parameter_soup_warm60_f{fold}.npz")
    x3d = load(ROOT / f"outputs/study_x3dl_soup/selected_cropblend50_fold{fold}.npz")
    auxiliary = load(ROOT / f"experiments/v36_auxiliary_rescue/outputs/v23_aux_adabn_f{fold}.npz")
    return {
        "experts": [
            anchor["probabilities"].astype(np.float32),
            aligned(indices, adapted),
            aligned(indices, base),
            aligned(indices, x3d),
            aligned(indices, auxiliary, "probabilities_skeleton"),
            aligned(indices, auxiliary, "probabilities_imu"),
        ],
        "indices": indices,
        "labels": anchor["labels"].astype(np.int64),
        "presence": presence_values[indices],
    }


def fold_domains(values, records, fold):
    return np.asarray(
        [
            int(records[int(index)]["user"] > 15)
            + 2 * int(records[int(index)]["user"] in PAIRED_USERS[fold])
            for index in values["indices"]
        ],
        dtype=np.int64,
    )


def subset(values, selected):
    return {
        "experts": [expert[selected] for expert in values["experts"]],
        "indices": values["indices"][selected],
        "labels": values["labels"][selected],
        "presence": values["presence"][selected],
    }


def raw_metrics(probabilities, labels):
    predictions = probabilities.argmax(axis=1)
    return {
        "correct": int((predictions == labels).sum()),
        "rows": len(labels),
        "accuracy": float((predictions == labels).mean()),
        "nll": float(
            -np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1)).mean()
        ),
    }


def cross_user_selection(fold0, domains, records):
    users = np.asarray([records[int(index)]["user"] for index in fold0["indices"]])
    candidates = []
    for retention_weight in RETENTION_WEIGHTS:
        probabilities = np.empty((len(fold0["indices"]), 40), dtype=np.float32)
        histories = {}
        for held_user in sorted(set(users.tolist())):
            train_mask = users != held_user
            valid_mask = users == held_user
            train = subset(fold0, train_mask)
            valid = subset(fold0, valid_mask)
            model, history = fit(
                train["experts"],
                train["labels"],
                train["indices"],
                domains[train_mask],
                train["presence"],
                records,
                retention_weight,
                steps=STEPS,
                seed=42 + held_user,
            )
            probabilities[valid_mask], _ = predict(
                model,
                valid["experts"],
                valid["indices"],
                domains[valid_mask],
                valid["presence"],
                records,
            )
            histories[str(held_user)] = history
        candidate = {
            "retention_weight": retention_weight,
            "metrics": raw_metrics(probabilities, fold0["labels"]),
            "history": histories,
        }
        candidates.append(candidate)
        print(json.dumps({key: value for key, value in candidate.items() if key != "history"}), flush=True)
    selected = max(
        candidates,
        key=lambda item: (
            item["metrics"]["correct"],
            -item["metrics"]["nll"],
            item["retention_weight"],
        ),
    )
    return selected, candidates


def train_predict(train, train_domains, valid, valid_domains, records, retention_weight, seed):
    model, history = fit(
        train["experts"],
        train["labels"],
        train["indices"],
        train_domains,
        train["presence"],
        records,
        retention_weight,
        steps=STEPS,
        seed=seed,
    )
    probabilities, sequence = predict(
        model,
        valid["experts"],
        valid["indices"],
        valid_domains,
        valid["presence"],
        records,
    )
    return model, probabilities, history, sequence


def decoded_metrics(fold, probabilities, values, records):
    return run(
        fold,
        (probabilities, values["indices"], values["labels"]),
        records,
        SETTINGS,
    )


def test_values(presence_values, records):
    output = ROOT / "experiments/v24_full_refit/outputs"
    anchor = load(output / "predictions/v52_domain_logit_residual_domain_gate.npz")
    indices = anchor["indices"].astype(np.int64)
    adapted = load(output / "predictions/v24_full_warm60_pseudo_adapt.npz")
    base = load(output / "predictions/v24_full_warm60_int8_tta.npz")
    x3d = load(ROOT / "outputs/predictions/v9_x3dl_soup_full_cropblend50.npz")
    auxiliary = load(output / "predictions/v37_auxiliary_adabn.npz")
    domains = []
    for index in indices:
        interval = record_interval(records[int(index)])
        date = datetime.fromtimestamp(interval[0]).strftime("%Y-%m-%d")
        if date not in SOURCE_BY_DATE:
            raise ValueError(f"Unexpected test acquisition date: {date}")
        domains.append(int(date >= "2025-06-09") + 2 * int(SOURCE_BY_DATE[date] is not None))
    return {
        "experts": [
            anchor["probabilities"].astype(np.float32),
            aligned(indices, adapted),
            aligned(indices, base),
            aligned(indices, x3d),
            aligned(indices, auxiliary, "probabilities_skeleton"),
            aligned(indices, auxiliary, "probabilities_imu"),
        ],
        "indices": indices,
        "presence": presence_values[indices],
        "domains": np.asarray(domains, dtype=np.int64),
    }


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    records = get_records()
    presence_values = modality_presence(MODALITIES)
    fold0 = load_fold(0, presence_values)
    fold1 = load_fold(1, presence_values)
    domains0 = fold_domains(fold0, records, 0)
    domains1 = fold_domains(fold1, records, 1)
    selected, selection_candidates = cross_user_selection(fold0, domains0, records)
    retention_weight = selected["retention_weight"]
    _, prediction1, history01, sequence1 = train_predict(
        fold0, domains0, fold1, domains1, records, retention_weight, 4201
    )
    blind_fold1 = {
        "raw": raw_metrics(prediction1, fold1["labels"]),
        "decoded": decoded_metrics(1, prediction1, fold1, records),
    }
    print(json.dumps({"blind_fold1": blind_fold1}, indent=2), flush=True)
    _, prediction0, history10, sequence0 = train_predict(
        fold1, domains1, fold0, domains0, records, retention_weight, 4210
    )
    predictions = (prediction0, prediction1)
    values = (fold0, fold1)
    v57 = [
        load(ROOT / f"experiments/v57_round_transformer_distillation/outputs/predictions_f{fold}.npz")
        for fold in (0, 1)
    ]
    aligned_v57 = [aligned(values[fold]["indices"], v57[fold]) for fold in (0, 1)]
    blend_candidates = []
    for weight in BLEND_WEIGHTS:
        probabilities = (1 - weight) * aligned_v57[0] + weight * predictions[0]
        blend_candidates.append(
            {
                "weight": weight,
                "raw": raw_metrics(probabilities, fold0["labels"]),
                "decoded": decoded_metrics(0, probabilities, fold0, records),
            }
        )
    selected_blend = max(
        blend_candidates,
        key=lambda item: (
            item["decoded"]["correct"],
            item["raw"]["correct"],
            -item["weight"],
        ),
    )
    weight = selected_blend["weight"]
    fold1_blend = (1 - weight) * aligned_v57[1] + weight * prediction1
    fold1_result = {
        "weight": weight,
        "raw": raw_metrics(fold1_blend, fold1["labels"]),
        "decoded": decoded_metrics(1, fold1_blend, fold1, records),
    }
    combined_correct = selected_blend["decoded"]["correct"] + fold1_result["decoded"]["correct"]
    output = HERE / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "crossfit_f0.npz", probabilities=prediction0, indices=fold0["indices"], labels=fold0["labels"])
    np.savez(output / "crossfit_f1.npz", probabilities=prediction1, indices=fold1["indices"], labels=fold1["labels"])
    train = {
        "experts": [np.concatenate([fold0["experts"][i], fold1["experts"][i]]) for i in range(6)],
        "indices": np.concatenate([fold0["indices"], fold1["indices"]]),
        "labels": np.concatenate([fold0["labels"], fold1["labels"]]),
        "presence": np.concatenate([fold0["presence"], fold1["presence"]]),
    }
    train_domains = np.concatenate([domains0, domains1])
    test = test_values(presence_values, records)
    final_model, final_history = fit(
        train["experts"],
        train["labels"],
        train["indices"],
        train_domains,
        train["presence"],
        records,
        retention_weight,
        steps=STEPS,
        seed=4266,
    )
    supervised_test, test_sequence = predict(
        final_model,
        test["experts"],
        test["indices"],
        test["domains"],
        test["presence"],
        records,
    )
    v57_test = load(
        ROOT / "experiments/v24_full_refit/outputs/predictions/v57_round_transformer_distilled.npz"
    )
    v57_test_probabilities = aligned(test["indices"], v57_test)
    hybrid_test = (1 - weight) * v57_test_probabilities + weight * supervised_test
    hybrid_path = output / "v66_hybrid_test.npz"
    np.savez(hybrid_path, probabilities=hybrid_test, indices=test["indices"])
    bundle = ROOT / "experiments/v24_full_refit/outputs/final/v66_supervised_expert_transformer.pt"
    save_bundle(final_model, bundle, retention_weight, STEPS)
    decoded = ROOT / "experiments/v24_full_refit/outputs/predictions/v66_supervised_expert_hybrid.npz"
    submission = ROOT / "experiments/v24_full_refit/outputs/submissions/v66_supervised_expert_hybrid.csv"
    decoder_report = ROOT / "experiments/v24_full_refit/outputs/reports/v66_supervised_expert_hybrid.json"
    subprocess.run(
        [
            str(Path.home() / "miniconda3/envs/Train/bin/python"),
            "-B",
            str(ROOT / "experiments/scene_context_v13/build_pair_aware_submission.py"),
            "--input",
            str(hybrid_path),
            "--output",
            str(decoded),
            "--submission",
            str(submission),
            "--report",
            str(decoder_report),
        ],
        check=True,
    )
    assets = [
        ROOT / "experiments/v24_full_refit/outputs/final/v24_full_warm60_soup_int8.pt",
        ROOT / "experiments/v24_full_refit/outputs/final/v24_full_warm60_pseudo_head_delta.pt",
        ROOT / "experiments/v24_full_refit/outputs/final/v52_domain_logit_residual_conv1d.pt",
        ROOT / "experiments/v24_full_refit/outputs/final/v57_round_residual_transformer.pt",
        ROOT / "outputs/final_legal_multimodal_v9/models/yolo11n_compact.pt",
        ROOT / "outputs/checkpoints/v8_x3dl_soup_full_w075/best_int8_compact.pt",
        ROOT / "outputs/final_legal_multimodal_v9/models/person_detector_int8_compact.pth",
        bundle,
    ]
    asset_bytes = sum(path.stat().st_size for path in assets)
    if asset_bytes >= 100_000_000:
        raise RuntimeError(f"V66 exceeds the model limit: {asset_bytes}")
    v59_submission = ROOT / "experiments/v24_full_refit/outputs/submissions/v59_teacher_sequence_hybrid.csv"
    with submission.open() as handle:
        next(handle)
        candidate_labels = np.asarray([int(row[1]) for row in csv.reader(handle)])
    with v59_submission.open() as handle:
        next(handle)
        v59_labels = np.asarray([int(row[1]) for row in csv.reader(handle)])
    report = {
        "protocol": (
            "Retention selected by Fold-0 leave-one-user-out validation; Fold 1 evaluated once blind. "
            "V57 blend weight selected on cross-fitted Fold 0 and transferred unchanged to Fold 1."
        ),
        "selection": selected,
        "selection_candidates": selection_candidates,
        "blind_fold1_standalone": blind_fold1,
        "crossfit": {
            "fold0": {
                "raw": raw_metrics(prediction0, fold0["labels"]),
                "decoded": decoded_metrics(0, prediction0, fold0, records),
                "sequence": sequence0,
            },
            "fold1": {
                "raw": raw_metrics(prediction1, fold1["labels"]),
                "decoded": decoded_metrics(1, prediction1, fold1, records),
                "sequence": sequence1,
            },
        },
        "selected_blend": selected_blend,
        "fold1_blend": fold1_result,
        "blend_candidates": blend_candidates,
        "combined": {"correct": combined_correct, "rows": 1357, "accuracy": combined_correct / 1357},
        "reference_v59": {"correct": 1197, "rows": 1357},
        "promote": (
            combined_correct > 1197
            and fold1_result["decoded"]["correct"] >= 628
            and int((candidate_labels != v59_labels).sum()) > 0
        ),
        "final_training": {"history": final_history, "test_sequence": test_sequence},
        "assets": [str(path) for path in assets],
        "asset_bytes": asset_bytes,
        "slack_bytes": 100_000_000 - asset_bytes,
        "submission": str(submission),
        "submission_sha256": sha256(submission),
        "changed_test_labels_vs_v59": int((candidate_labels != v59_labels).sum()),
    }
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("selection_candidates", "blend_candidates")}, indent=2))


if __name__ == "__main__":
    main()


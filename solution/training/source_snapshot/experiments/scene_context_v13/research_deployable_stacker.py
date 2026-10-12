#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.blend_aware_v12.audit_existing import align, load
from experiments.scene_context_v13.research_sequence_decoder import selected_probabilities
from scripts.evaluate_repeat_decoder import decode, transition_matrix
from src.data import FOLDS, get_records


def components(fold):
    reference = load(ROOT / f"outputs/study_r2plus1d/lightweight_base_int8/base_fold{fold}.npz")
    mixup_name = "v12_mixup_f0_selected_raw_int8.npz" if fold == 0 else "v12_mixup_f1_selected_raw_int8.npz"
    probabilities = [
        reference["probabilities"],
        align(
            reference,
            load(ROOT / f"experiments/blend_aware_v12/outputs/predictions/v12_lowrank04_old_f{fold}.npz"),
        ),
        align(
            reference,
            load(ROOT / "experiments/blend_aware_v12/outputs/predictions" / mixup_name),
        ),
        align(
            reference,
            load(ROOT / f"outputs/study_x3dl_soup/selected_cropblend50_fold{fold}.npz"),
        ),
    ]
    final = selected_probabilities(fold)[0]
    return reference, probabilities, final


def features(probabilities):
    output = []
    for values in probabilities:
        logits = np.log(np.clip(values, 1e-8, 1.0))
        logits -= logits.mean(axis=1, keepdims=True)
        output.append(logits)
    return np.concatenate(output, axis=1)


def fit_predict(train_x, train_y, valid_x, regularization, class_weight):
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    valid_x = scaler.transform(valid_x)
    model = LogisticRegression(
        C=regularization,
        class_weight=class_weight,
        max_iter=2000,
        solver="lbfgs",
    )
    model.fit(train_x, train_y)
    probabilities = np.full((len(valid_x), 40), 1e-8, dtype=np.float64)
    probabilities[:, model.classes_.astype(int)] = model.predict_proba(valid_x)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def decoder_metrics(probabilities, reference, fold, records):
    transitions = [
        transition_matrix(records, set(FOLDS[fold]), environment=environment)
        for environment in (0, 1)
    ]
    result = decode(
        probabilities,
        reference["indices"],
        records,
        maximum_gap=120,
        minimum_similarity=0.7,
        enforce_unique=True,
        transition_strength=0.4,
        environment_transitions=transitions,
        repeat_pool_power=1.5,
    )
    return {
        "raw_correct": int((probabilities.argmax(1) == reference["labels"]).sum()),
        "raw_accuracy": float((probabilities.argmax(1) == reference["labels"]).mean()),
        "decoded_correct": round(result["accuracy"] * len(reference["labels"])),
        "decoded_accuracy": result["accuracy"],
    }


def main():
    records = get_records()
    folds = [components(fold) for fold in (0, 1)]
    candidates = []
    for source_fold, target_fold in ((0, 1), (1, 0)):
        source, source_probabilities, source_final = folds[source_fold]
        target, target_probabilities, target_final = folds[target_fold]
        source_x = features(source_probabilities)
        target_x = features(target_probabilities)
        source_users = np.asarray([records[int(index)]["user"] for index in source["indices"]])
        settings = []
        for regularization in (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1):
            for class_weight in (None, "balanced"):
                stacked_oof = np.zeros_like(source_final)
                for user in np.unique(source_users):
                    train = source_users != user
                    valid = ~train
                    stacked_oof[valid] = fit_predict(
                        source_x[train],
                        source["labels"][train],
                        source_x[valid],
                        regularization,
                        class_weight,
                    )
                for alpha in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
                    blended = (1 - alpha) * source_final + alpha * stacked_oof
                    metrics = decoder_metrics(blended, source, source_fold, records)
                    settings.append(
                        {
                            "regularization": regularization,
                            "class_weight": class_weight,
                            "alpha": alpha,
                            "source_leave_user": metrics,
                        }
                    )
        selected = max(
            settings,
            key=lambda item: (
                item["source_leave_user"]["decoded_correct"],
                item["source_leave_user"]["raw_correct"],
                -item["alpha"],
                -item["regularization"],
                item["class_weight"] is None,
            ),
        )
        stacked = fit_predict(
            source_x,
            source["labels"],
            target_x,
            selected["regularization"],
            selected["class_weight"],
        )
        blended = (1 - selected["alpha"]) * target_final + selected["alpha"] * stacked
        candidates.append(
            {
                "source_fold": source_fold,
                "target_fold": target_fold,
                "selected": selected,
                "target": decoder_metrics(blended, target, target_fold, records),
                "target_baseline": decoder_metrics(target_final, target, target_fold, records),
            }
        )
    report = {
        "protocol": "Stacker uses only four already deployed experts; hyperparameters selected by leave-one-user-out stacking inside the source fold, then evaluated on the opposite fold.",
        "directions": candidates,
    }
    output = ROOT / "experiments/scene_context_v13/outputs/deployable_stacker.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

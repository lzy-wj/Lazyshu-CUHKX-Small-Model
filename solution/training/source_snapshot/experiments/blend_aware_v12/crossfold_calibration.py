#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_repeat_decoder import decode, transition_matrix
from src.data import FOLDS, get_records


PENALTIES = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)


def softmax(logits):
    logits = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def fit_bias(probabilities, labels, penalty):
    logits = np.log(np.clip(probabilities, 1e-12, 1.0))

    def objective(bias):
        centered = bias - bias.mean()
        calibrated = softmax(logits + centered)
        loss = -np.log(np.clip(calibrated[np.arange(len(labels)), labels], 1e-12, 1)).mean()
        gradient = calibrated
        gradient[np.arange(len(labels)), labels] -= 1
        gradient = gradient.mean(axis=0)
        regularization = penalty * np.mean(centered ** 2)
        regularization_gradient = 2 * penalty * centered / len(centered)
        return loss + regularization, gradient + regularization_gradient

    result = minimize(objective, np.zeros(40), jac=True, method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(result.message)
    return result.x - result.x.mean()


def nll(probabilities, labels):
    return float(-np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1)).mean())


def main():
    records = get_records()
    folds = []
    for fold in (0, 1):
        with np.load(
            ROOT / f"outputs/study_r2plus1d/v9_l_cropblend50_gated/fold{fold}.npz",
            allow_pickle=False,
        ) as values:
            folds.append({key: values[key] for key in values.files})
    results = []
    calibrated_folds = []
    for target_fold in (0, 1):
        source = folds[1 - target_fold]
        source_users = np.asarray([records[int(index)]["user"] for index in source["indices"]])
        validation = []
        for penalty in PENALTIES:
            losses = []
            for user in np.unique(source_users):
                train = source_users != user
                valid = ~train
                bias = fit_bias(source["probabilities"][train], source["labels"][train], penalty)
                probabilities = softmax(
                    np.log(np.clip(source["probabilities"][valid], 1e-12, 1.0)) + bias
                )
                losses.append(nll(probabilities, source["labels"][valid]))
            validation.append((float(np.mean(losses)), penalty))
        _, selected_penalty = min(validation)
        bias = fit_bias(source["probabilities"], source["labels"], selected_penalty)
        target = folds[target_fold]
        probabilities = softmax(np.log(np.clip(target["probabilities"], 1e-12, 1.0)) + bias)
        transitions = [
            transition_matrix(records, set(FOLDS[target_fold]), environment=environment)
            for environment in (0, 1)
        ]
        decoded = decode(
            probabilities,
            target["indices"],
            records,
            maximum_gap=120,
            minimum_similarity=0.7,
            enforce_unique=True,
            transition_strength=0.4,
            environment_transitions=transitions,
            repeat_pool_power=2,
        )
        metrics = {
            "fold": target_fold,
            "source_fold": 1 - target_fold,
            "selected_penalty": selected_penalty,
            "source_leave_user_nll": dict((str(penalty), loss) for loss, penalty in validation),
            "raw_accuracy": float((probabilities.argmax(axis=1) == target["labels"]).mean()),
            "nll": nll(probabilities, target["labels"]),
            "decoded_accuracy": float(decoded["accuracy"]),
            "decoded_correct": round(float(decoded["accuracy"]) * len(target["labels"])),
            "maximum_absolute_bias": float(np.abs(bias).max()),
        }
        results.append(metrics)
        calibrated_folds.append(probabilities)
    report = {
        "protocol": "Each target fold receives class-logit biases fit on the opposite fold; penalty is selected by leave-one-user-out NLL inside the source fold.",
        "folds": results,
        "weighted_decoded": sum(item["decoded_correct"] for item in results) / 1357,
        "v9_weighted_decoded": 1095 / 1357,
    }
    output = ROOT / "experiments/blend_aware_v12/outputs/crossfold_calibration.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    for fold, probabilities in enumerate(calibrated_folds):
        np.savez(
            output.with_name(f"crossfold_calibrated_fold{fold}.npz"),
            probabilities=probabilities,
            labels=folds[fold]["labels"],
            indices=folds[fold]["indices"],
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

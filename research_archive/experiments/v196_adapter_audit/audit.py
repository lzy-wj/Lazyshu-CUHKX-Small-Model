#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUNS = {
    "v187_mixup_s42_lr1e3": ROOT / "experiments/v187_statistics_only_adapter/outputs/checkpoints/v187_statistics_only_f{fold}",
    "v189_mixup_s42_lr3e4": ROOT / "experiments/v189_statistics_adapter_lowlr/outputs/checkpoints/v189_statistics_lowlr_f{fold}",
    "v193_clean_s42_lr1e3": ROOT / "experiments/v193_statistics_token_clean/outputs/checkpoints/v193_statistics_token_clean_f{fold}",
    "v194_mixup_s43_lr1e3": ROOT / "experiments/v194_statistics_token_seed43/outputs/checkpoints/v194_statistics_token_seed43_f{fold}",
    "v197_device_attention_s42": ROOT / "experiments/v197_device_attention_adapter/outputs/checkpoints/v197_device_attention_s42_f{fold}",
    "v197_device_attention_s43": ROOT / "experiments/v197_device_attention_adapter/outputs/checkpoints/v197_device_attention_s43_f{fold}",
}


def source_path(fold):
    return ROOT / f"experiments/scene_context_v13/outputs/checkpoints/v23_shared5_joint_warm_maskdistill_f{fold}/validation.npz"


def load_aligned(path, indices):
    values = np.load(path)
    lookup = {int(index): position for position, index in enumerate(values["indices"])}
    order = np.asarray([lookup[int(index)] for index in indices])
    return values["probabilities"][order]


def metrics(probabilities, labels, indices, source_prediction):
    prediction = probabilities.argmax(1)
    changed = prediction != source_prediction
    return {
        "correct": int((prediction == labels).sum()),
        "rows": len(labels),
        "accuracy": float((prediction == labels).mean()),
        "nll": float(-np.log(probabilities[np.arange(len(labels)), labels].clip(1e-8)).mean()),
        "changes": int(changed.sum()),
        "gains": int((changed & (prediction == labels) & (source_prediction != labels)).sum()),
        "losses": int((changed & (prediction != labels) & (source_prediction == labels)).sum()),
        "changed_indices": [int(index) for index in indices[changed]],
    }


def fold_data(fold):
    source = np.load(source_path(fold))
    indices = source["indices"].astype(np.int64)
    labels = source["labels"].astype(np.int64)
    probabilities = source["probabilities"]
    prediction = probabilities.argmax(1)
    return indices, labels, probabilities, prediction


def run_epochs(template, fold):
    directory = Path(str(template).format(fold=fold)) / "epochs"
    return {
        int(path.stem.split("_")[-1]): path
        for path in sorted(directory.glob("validation_*.npz"))
    }


def main():
    report = {
        "protocol": "Select epoch on Fold0 by correct, losses, changes, and NLL; transfer identical epoch to Fold1.",
        "runs": {},
    }
    for name, template in RUNS.items():
        folds = {}
        for fold in (0, 1):
            indices, labels, source_probabilities, source_prediction = fold_data(fold)
            source_metrics = metrics(source_probabilities, labels, indices, source_prediction)
            epochs = {
                str(epoch): metrics(load_aligned(path, indices), labels, indices, source_prediction)
                for epoch, path in run_epochs(template, fold).items()
            }
            folds[str(fold)] = {"source": source_metrics, "epochs": epochs}
        common_epochs = sorted(set(map(int, folds["0"]["epochs"])) & set(map(int, folds["1"]["epochs"])))
        if not common_epochs:
            report["runs"][name] = {"folds": folds, "status": "no_common_epoch"}
            continue
        selected_epoch = max(
            common_epochs,
            key=lambda epoch: (
                folds["0"]["epochs"][str(epoch)]["correct"],
                -folds["0"]["epochs"][str(epoch)]["losses"],
                -folds["0"]["epochs"][str(epoch)]["changes"],
                -folds["0"]["epochs"][str(epoch)]["nll"],
                -epoch,
            ),
        )
        selected = {fold: folds[fold]["epochs"][str(selected_epoch)] for fold in ("0", "1")}
        source_total = sum(folds[fold]["source"]["correct"] for fold in ("0", "1"))
        selected_total = sum(selected[fold]["correct"] for fold in ("0", "1"))
        optimistic = {
            fold: max(folds[fold]["epochs"].items(), key=lambda item: (item[1]["correct"], -item[1]["nll"]))
            for fold in ("0", "1")
        }
        report["runs"][name] = {
            "folds": folds,
            "selected_epoch": selected_epoch,
            "selected": selected,
            "source_total": source_total,
            "selected_total": selected_total,
            "delta": selected_total - source_total,
            "optimistic_per_fold_upper_bound": {
                fold: {"epoch": int(epoch), "metrics": values}
                for fold, (epoch, values) in optimistic.items()
            },
        }
    destination = HERE / "outputs/audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    concise = {
        name: {
            "selected_epoch": values.get("selected_epoch"),
            "fold0": values.get("selected", {}).get("0", {}).get("correct"),
            "fold1": values.get("selected", {}).get("1", {}).get("correct"),
            "delta": values.get("delta"),
        }
        for name, values in report["runs"].items()
    }
    print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    main()

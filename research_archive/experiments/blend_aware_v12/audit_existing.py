#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_repeat_decoder import decode, transition_matrix
from src.data import FOLDS, get_records


def load(path):
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key] for key in values.files}


def align(reference, candidate, key="probabilities"):
    positions = {int(index): position for position, index in enumerate(candidate["indices"])}
    try:
        order = np.asarray([positions[int(index)] for index in reference["indices"]])
    except KeyError:
        return None
    if "labels" in candidate and not np.array_equal(reference["labels"], candidate["labels"][order]):
        return None
    probabilities = candidate.get(key)
    if probabilities is None or probabilities.ndim != 2 or probabilities.shape[1] != 40:
        return None
    return probabilities[order].astype(np.float64)


def temperature_scale(probabilities, temperature):
    logits = np.log(np.clip(probabilities, 1e-12, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def accuracy(labels, probabilities):
    return float((probabilities.argmax(axis=1) == labels).mean())


def apply_auxiliary(probabilities, auxiliary):
    output = probabilities.copy()
    selected = output.max(axis=1) < 0.15
    output[selected] = 0.9 * output[selected] + 0.1 * auxiliary[selected]
    return output


def candidate_pairs():
    pairs = []
    checkpoints = ROOT / "outputs/checkpoints"
    for fold0 in sorted(checkpoints.glob("*_f0/validation.npz")):
        stem = fold0.parent.name[:-3]
        fold1 = checkpoints / f"{stem}_f1/validation.npz"
        if fold1.is_file():
            pairs.append((stem, fold0, fold1, "probabilities"))
    special = [
        (
            "r2_soup_auxiliary_video",
            ROOT / "outputs/study_r2plus1d/r2_soup_auxiliary/fold0.npz",
            ROOT / "outputs/study_r2plus1d/r2_soup_auxiliary/fold1.npz",
            "probabilities_video",
        ),
        (
            "r2_soup_auxiliary_skeleton",
            ROOT / "outputs/study_r2plus1d/r2_soup_auxiliary/fold0.npz",
            ROOT / "outputs/study_r2plus1d/r2_soup_auxiliary/fold1.npz",
            "probabilities_skeleton",
        ),
        (
            "r2_soup_auxiliary_imu",
            ROOT / "outputs/study_r2plus1d/r2_soup_auxiliary/fold0.npz",
            ROOT / "outputs/study_r2plus1d/r2_soup_auxiliary/fold1.npz",
            "probabilities_imu",
        ),
    ]
    return pairs + [pair for pair in special if pair[1].is_file() and pair[2].is_file()]


def main():
    references = [
        load(ROOT / f"outputs/study_r2plus1d/lightweight_base_int8/base_fold{fold}.npz")
        for fold in (0, 1)
    ]
    r2 = [
        align(
            references[fold],
            load(ROOT / f"outputs/study_r2plus1d/r2_staged_soup_f{fold}_w075_crop075.npz"),
        )
        for fold in (0, 1)
    ]
    auxiliary = [
        align(
            references[fold],
            load(ROOT / f"outputs/study_x3dl_soup/selected_cropblend50_fold{fold}.npz"),
        )
        for fold in (0, 1)
    ]
    v9_main = [
        0.4 * references[fold]["probabilities"] + 0.6 * temperature_scale(r2[fold], 2.0)
        for fold in (0, 1)
    ]
    v9 = [apply_auxiliary(v9_main[fold], auxiliary[fold]) for fold in (0, 1)]
    baseline_raw = [accuracy(references[fold]["labels"], v9[fold]) for fold in (0, 1)]

    results = []
    temperatures = (0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
    weights = np.linspace(0.025, 0.3, 12)
    for name, fold0_path, fold1_path, key in candidate_pairs():
        candidates = []
        for fold, path in enumerate((fold0_path, fold1_path)):
            candidates.append(align(references[fold], load(path), key=key))
        if any(probabilities is None for probabilities in candidates):
            continue
        grid = []
        for temperature in temperatures:
            scaled = temperature_scale(candidates[0], temperature)
            for weight in weights:
                probability = apply_auxiliary(
                    (1.0 - weight) * v9_main[0] + weight * scaled,
                    auxiliary[0],
                )
                grid.append((accuracy(references[0]["labels"], probability), weight, temperature))
        selected_raw, weight, temperature = max(grid, key=lambda item: (item[0], -item[1]))
        fold_metrics = []
        for fold in (0, 1):
            probability = apply_auxiliary(
                (1.0 - weight) * v9_main[fold]
                + weight * temperature_scale(candidates[fold], temperature),
                auxiliary[fold],
            )
            raw = accuracy(references[fold]["labels"], probability)
            fold_metrics.append(
                {
                    "standalone": accuracy(references[fold]["labels"], candidates[fold]),
                    "raw": raw,
                    "raw_delta": raw - baseline_raw[fold],
                }
            )
        results.append(
            {
                "name": name,
                "paths": [str(fold0_path.relative_to(ROOT)), str(fold1_path.relative_to(ROOT))],
                "key": key,
                "weight": float(weight),
                "temperature": temperature,
                "folds": fold_metrics,
                "minimum_raw_delta": min(item["raw_delta"] for item in fold_metrics),
                "weighted_raw_delta": (
                    652 * fold_metrics[0]["raw_delta"] + 705 * fold_metrics[1]["raw_delta"]
                ) / 1357,
            }
        )

    results.sort(key=lambda item: (item["minimum_raw_delta"], item["weighted_raw_delta"]), reverse=True)
    records = get_records()
    for result in results[:12]:
        decoded = []
        for fold in (0, 1):
            candidate = align(references[fold], load(ROOT / result["paths"][fold]), key=result["key"])
            probability = apply_auxiliary(
                (1.0 - result["weight"]) * v9_main[fold]
                + result["weight"] * temperature_scale(candidate, result["temperature"]),
                auxiliary[fold],
            )
            transitions = [
                transition_matrix(records, set(FOLDS[fold]), environment=environment)
                for environment in (0, 1)
            ]
            metrics = decode(
                probability,
                references[fold]["indices"],
                records,
                maximum_gap=120,
                minimum_similarity=0.7,
                enforce_unique=True,
                transition_strength=0.4,
                environment_transitions=transitions,
                repeat_pool_power=2,
            )
            decoded.append(float(metrics["accuracy"]))
        result["decoded"] = decoded
        result["decoded_correct"] = [round(decoded[0] * 652), round(decoded[1] * 705)]
        result["weighted_decoded"] = sum(result["decoded_correct"]) / 1357

    output = ROOT / "experiments/blend_aware_v12/outputs/existing_candidates.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol": "Select scalar weight and temperature on fold 0 raw accuracy; fold 1 is untouched confirmation.",
        "v9_raw": baseline_raw,
        "v9_decoded_correct": [501, 594],
        "v9_weighted_decoded": 1095 / 1357,
        "candidate_count": len(results),
        "results": results,
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report | {"results": results[:12]}, indent=2))


if __name__ == "__main__":
    main()

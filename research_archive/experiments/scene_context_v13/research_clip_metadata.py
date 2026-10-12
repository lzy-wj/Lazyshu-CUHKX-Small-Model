#!/usr/bin/env python3
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.scene_context_v13.research_sequence_decoder import selected_probabilities
from scripts.evaluate_repeat_decoder import decode, transition_matrix
from src.data import FOLDS, get_records


TIMESTAMP = re.compile(r"(20\d\d-\d\d-\d\d[_ ]\d\d[-:]\d\d[-:]\d\d\.\d+)")


def file_summary(folder):
    folder = Path(folder)
    if not folder.is_dir():
        return [0.0] * 5
    files = [path for path in folder.iterdir() if path.is_file()]
    sizes = np.asarray([path.stat().st_size for path in files], dtype=np.float64)
    timestamps = []
    for path in files:
        match = TIMESTAMP.search(path.name)
        if match:
            try:
                timestamps.append(
                    datetime.strptime(
                        match.group(1).replace(" ", "_").replace(":", "-"),
                        "%Y-%m-%d_%H-%M-%S.%f",
                    ).timestamp()
                )
            except ValueError:
                pass
    duration = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0.0
    return [
        np.log1p(len(files)),
        np.log1p(sizes.sum()) if len(sizes) else 0.0,
        np.log1p(np.median(sizes)) if len(sizes) else 0.0,
        np.log1p(duration),
        np.log1p(len(files) / max(duration, 1e-3)),
    ]


def build_features(records):
    cache = ROOT / "experiments/scene_context_v13/outputs/clip_metadata_features.npy"
    if cache.is_file():
        values = np.load(cache)
        if values.shape[0] == len(records):
            return values
    boxes = np.load(ROOT / "data/metadata/person_boxes.npz")
    sensor_mask = np.load(ROOT / "data/processed/sensor_mask.npy")
    video_mask = np.load(ROOT / "data/processed/video_mask.npy")
    imu_timed = np.load(ROOT / "data/processed/imu_timed.npy", mmap_mode="r")
    radar_frames = np.load(ROOT / "data/processed/radar_points_frame_count.npy")
    motion = json.load(open(ROOT / "data/metadata/thermal_motion.json"))["clips_detail"]
    rows = []
    for record in records:
        features = []
        for modality in ("Depth_Color", "IR", "Thermal", "Skeleton", "IMU", "Radar"):
            features.extend(file_summary(record["modalities"].get(modality, "")))
        index = record["index"]
        features.extend(sensor_mask[index].tolist())
        features.extend(video_mask[index].tolist())
        for modality in range(2):
            left, top, right, bottom = boxes["boxes"][index, modality]
            features.extend(
                [
                    boxes["confidence"][index, modality],
                    right - left,
                    bottom - top,
                    (right - left) * (bottom - top),
                    (left + right) / 2,
                    (top + bottom) / 2,
                ]
            )
        duration_channels = [18, 38, 58, 78, 98]
        features.extend(imu_timed[index, 0, duration_channels].astype(float).tolist())
        features.append(np.log1p(radar_frames[index]))
        left, top, right, bottom = motion[index]["box"]
        features.extend([right - left, bottom - top, (right - left) * (bottom - top)])
        rows.append(features)
    values = np.asarray(rows, dtype=np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, values)
    return values


def decoder_metrics(probabilities, indices, labels, records, fold):
    transitions = [
        transition_matrix(records, set(FOLDS[fold]), environment=environment)
        for environment in (0, 1)
    ]
    result = decode(
        probabilities,
        indices,
        records,
        maximum_gap=120,
        minimum_similarity=0.7,
        enforce_unique=True,
        transition_strength=0.4,
        environment_transitions=transitions,
        repeat_pool_power=1.5,
    )
    return {
        "raw_correct": int((probabilities.argmax(1) == labels).sum()),
        "decoded_correct": round(result["accuracy"] * len(labels)),
        "decoded_accuracy": result["accuracy"],
    }


def main():
    records = get_records()
    features = build_features(records)
    settings = []
    for model_name in ("extra", "forest"):
        for leaf in (1, 2, 4, 8):
            fold_probabilities = []
            standalone = []
            for fold in (0, 1):
                train = np.asarray(
                    [
                        record["index"]
                        for record in records
                        if record["split"] == "train" and record["user"] not in FOLDS[fold]
                    ]
                )
                probabilities, indices, labels = selected_probabilities(fold)
                estimator = (
                    ExtraTreesClassifier(
                        n_estimators=500,
                        min_samples_leaf=leaf,
                        max_features=0.7,
                        class_weight="balanced",
                        n_jobs=12,
                        random_state=42,
                    )
                    if model_name == "extra"
                    else RandomForestClassifier(
                        n_estimators=500,
                        min_samples_leaf=leaf,
                        max_features=0.7,
                        class_weight="balanced_subsample",
                        n_jobs=12,
                        random_state=42,
                    )
                )
                model = make_pipeline(SimpleImputer(strategy="median"), estimator)
                model.fit(features[train], np.asarray([records[index]["label"] for index in train]))
                current = np.full((len(indices), 40), 1e-8, dtype=np.float64)
                predicted = model.predict_proba(features[indices])
                current[:, model[-1].classes_.astype(int)] = predicted
                current /= current.sum(axis=1, keepdims=True)
                fold_probabilities.append(current)
                standalone.append(float((current.argmax(1) == labels).mean()))
            settings.append(
                {
                    "model": model_name,
                    "leaf": leaf,
                    "probabilities": fold_probabilities,
                    "standalone": standalone,
                }
            )
    candidates = []
    fold0 = selected_probabilities(0)
    fold1 = selected_probabilities(1)
    for setting in settings:
        for temperature in (0.5, 0.75, 1.0, 1.5, 2.0):
            scaled = []
            for values in setting["probabilities"]:
                logits = np.log(np.clip(values, 1e-8, 1)) / temperature
                logits -= logits.max(axis=1, keepdims=True)
                current = np.exp(logits)
                scaled.append(current / current.sum(axis=1, keepdims=True))
            for weight in np.arange(0, 0.31, 0.025):
                probability = (1 - weight) * fold0[0] + weight * scaled[0]
                candidates.append(
                    {
                        "model": setting["model"],
                        "leaf": setting["leaf"],
                        "standalone": setting["standalone"],
                        "temperature": temperature,
                        "weight": float(weight),
                        "fold0_raw_correct": int((probability.argmax(1) == fold0[2]).sum()),
                        "scaled": scaled,
                    }
                )
    candidates.sort(key=lambda item: item["fold0_raw_correct"], reverse=True)
    for item in candidates[:80]:
        probability = (1 - item["weight"]) * fold0[0] + item["weight"] * item["scaled"][0]
        item["fold0"] = decoder_metrics(probability, fold0[1], fold0[2], records, 0)
    selected = max(
        candidates[:80],
        key=lambda item: (item["fold0"]["decoded_correct"], item["fold0"]["raw_correct"], -item["weight"]),
    )
    probability = (1 - selected["weight"]) * fold1[0] + selected["weight"] * selected["scaled"][1]
    selected["fold1"] = decoder_metrics(probability, fold1[1], fold1[2], records, 1)
    selected.pop("scaled")
    report = {
        "protocol": "Tree ensembles are diagnostic probes only. Model and blend selected on fold0; fold1 blind.",
        "feature_shape": list(features.shape),
        "selected": selected,
        "combined_correct": selected["fold0"]["decoded_correct"] + selected["fold1"]["decoded_correct"],
        "standalone_settings": [
            {key: value for key, value in item.items() if key != "probabilities"} for item in settings
        ],
    }
    output = ROOT / "experiments/scene_context_v13/outputs/clip_metadata_probe.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

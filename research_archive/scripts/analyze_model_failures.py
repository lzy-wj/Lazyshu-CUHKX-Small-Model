import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ROOT, get_records
from src.scratch_protocol import classification_metrics


def timestamp(path):
    match = re.search(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.\d+)", path.name)
    return pd.to_datetime(match.group(1), format="%Y-%m-%d_%H-%M-%S.%f") if match else None


def quantiles(values):
    return np.percentile(values, [0, 25, 50, 75, 100]).tolist() if values else []


def main():
    records = get_records()
    names = {int(path.name.split("_", 1)[0]): path.name.split("_", 1)[1]
             for path in (ROOT / "data/raw/HAR/data/Skeleton").iterdir() if path.is_dir()}
    indices, labels, probabilities = [], [], []
    for fold in [0, 1]:
        with (np.load(ROOT / f"outputs/checkpoints/fusion_input_v1_staged_f{fold}/validation.npz") as first,
              np.load(ROOT / f"outputs/checkpoints/fusion_input_v1_staged_s43_f{fold}/validation.npz") as second):
            np.testing.assert_array_equal(first["indices"], second["indices"])
            indices.extend(first["indices"])
            labels.extend(first["labels"])
            probabilities.extend((first["probabilities"] + second["probabilities"]) / 2)
    indices, labels, probability = np.asarray(indices), np.asarray(labels), np.asarray(probabilities)
    prediction = probability.argmax(1)
    metrics = classification_metrics(labels, prediction)
    confusion = np.asarray(metrics["confusion_matrix"])
    errors = confusion.copy()
    np.fill_diagonal(errors, 0)
    ordered = sorted(zip(*np.nonzero(errors)), key=lambda pair: -errors[pair])[:20]
    pairs = [{"truth": names[truth], "predicted": names[guessed], "count": int(errors[truth, guessed])} for truth, guessed in ordered]
    classes = []
    for label in range(40):
        training = [record for record in records if record["split"] == "train" and record["label"] == label]
        classes.append({"label": label, "name": names[label], "training_clips": len(training),
                        "training_users": len({record["user"] for record in training}),
                        "validation_support": metrics["class_support"][label], "recall": metrics["class_recall"][label]})
    body = (labels >= 28) & (labels <= 36)
    groups = {name: classification_metrics(labels[mask], prediction[mask])
              for name, mask in [("whole_body_actions_28_to_36", body), ("other_31_classes", ~body)]}
    timing = []
    for record in [record for record in records if record["split"] == "train"][::15]:
        if "IMU" not in record["modalities"] or "IR" not in record["modalities"]:
            continue
        stamps = [timestamp(path) for path in (ROOT / record["modalities"]["IR"]).iterdir() if path.suffix.lower() in [".png", ".jpg"]]
        stamps = [stamp for stamp in stamps if stamp is not None]
        tables = [pd.read_csv(path) for path in (ROOT / record["modalities"]["IMU"]).glob("*.csv")]
        if len(stamps) < 2 or not tables:
            continue
        table = pd.concat(tables, ignore_index=True)
        if table.empty:
            continue
        times = pd.to_datetime(table.iloc[:, 0], errors="coerce")
        starts, ends, counts = [], [], []
        for device in ["WTLA", "WTRA", "WTC", "WTLL", "WTRL"]:
            subset = times[table.iloc[:, 1].astype(str).str.startswith(device + "(")].dropna().drop_duplicates()
            if not subset.empty:
                starts.append(subset.min())
                ends.append(subset.max())
                counts.append(len(subset))
        if not starts:
            continue
        duration = (max(stamps) - min(stamps)).total_seconds()
        if duration <= 0:
            continue
        timing.append({"index": record["index"], "ir_seconds": duration, "ir_frames": len(stamps), "imu_counts": counts,
                       "device_start_spread": (max(starts) - min(starts)).total_seconds(),
                       "device_end_spread": (max(ends) - min(ends)).total_seconds(),
                       "imu_ir_start_offset": (min(starts) - min(stamps)).total_seconds(),
                       "imu_ir_end_offset": (max(ends) - max(stamps)).total_seconds()})
    starts = [row["device_start_spread"] for row in timing]
    ends = [row["device_end_spread"] for row in timing]
    sparse = [count for row in timing for count in row["imu_counts"]]
    report = {"status": "complete", "model": "Final two-seed development-fold ensemble, not the full-data model on its training labels",
              "accuracy": metrics["accuracy"], "balanced_accuracy": metrics["balanced_accuracy"], "macro_f1": metrics["macro_f1"],
              "top_confusions": pairs, "classes": classes, "action_groups": groups,
              "timestamp_audit": {"clips": len(timing), "device_start_spread_seconds": quantiles(starts), "device_end_spread_seconds": quantiles(ends),
                                  "imu_ir_start_offset_seconds": quantiles([row["imu_ir_start_offset"] for row in timing]),
                                  "imu_ir_end_offset_seconds": quantiles([row["imu_ir_end_offset"] for row in timing]),
                                  "device_observation_count": quantiles(sparse), "fraction_devices_under_64_observations": float(np.mean(np.asarray(sparse) < 64)),
                                  "relative_start_spread": quantiles([row["device_start_spread"] / row["ir_seconds"] for row in timing]),
                                  "details": timing,
                                  "caveat": "Every 15th labeled clip; recorded timestamps may require clock calibration. Window offsets are not automatically physical synchronization errors."}}
    destination = ROOT / "outputs/model_analysis"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "errors_and_timing.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Overall", {key: report[key] for key in ["accuracy", "balanced_accuracy", "macro_f1"]})
    print("Groups", {name: {key: value[key] for key in ["accuracy", "balanced_accuracy"]} for name, value in groups.items()})
    print("Top confusions", pairs[:10])
    print("Lowest recall", sorted(classes, key=lambda row: row["recall"])[:12])
    print("Timing", {key: value for key, value in report["timestamp_audit"].items() if key != "details"})


if __name__ == "__main__":
    main()

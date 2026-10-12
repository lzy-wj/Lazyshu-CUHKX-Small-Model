import argparse
import csv
import json
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
META = ROOT / "data/metadata"
CACHE = ROOT / "data/processed"
MODALITIES = ["Skeleton", "IMU", "Radar", "Depth_Color", "IR", "Thermal"]
DEVICES = ["WTLA", "WTRA", "WTC", "WTLL", "WTRL"]
PARENTS = [0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]


def interpolate(values, steps=64):
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return np.zeros((steps, values.shape[-1]), dtype=np.float32)
    values = np.nan_to_num(values, nan=0, posinf=0, neginf=0)
    source = np.linspace(0, 1, len(values))
    target = np.linspace(0, 1, steps)
    return np.stack([np.interp(target, source, values[:, channel])
                     for channel in range(values.shape[1])], axis=1).astype(np.float32)


def build_index():
    records = {}
    for modality in MODALITIES:
        for action in sorted((RAW / "HAR/data" / modality).glob("*")):
            if not action.is_dir() or not re.match(r"^\d+_", action.name):
                continue
            label = int(action.name.split("_", 1)[0])
            for subject in sorted(action.glob("user*")):
                user = int(subject.name.removeprefix("user"))
                for trial in sorted(subject.iterdir()):
                    if not trial.is_dir():
                        continue
                    key = f"train/{label}/{user}/{trial.name}"
                    record = records.setdefault(key, {
                        "id": key, "split": "train", "label": label, "user": user,
                        "trial": trial.name, "modalities": {},
                    })
                    record["modalities"][modality] = str(trial.relative_to(ROOT))
    source = ROOT / "data/downloads/CUHK-X_Small_Model_Track/Small-Model-Track/Testing/test_file/test.csv"
    with source.open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            folder = RAW / row["path"]
            if not folder.is_dir():
                raise RuntimeError(f"Missing test folder: {folder}")
            records[row["path"]] = {
                "id": row["path"], "split": "test", "label": -1, "user": -1,
                "trial": folder.name,
                "modalities": {name: str((folder / name).relative_to(ROOT))
                               for name in MODALITIES if (folder / name).is_dir()},
            }
    result = list(records.values())
    allowed = set(range(1, 10)) | set(range(16, 25))
    observed = {record["user"] for record in result if record["split"] == "train"}
    if not observed <= allowed:
        raise RuntimeError(f"Unexpected training subjects: {observed - allowed}")
    if len([record for record in result if record["split"] == "test"]) != 405:
        raise RuntimeError("Expected exactly 405 test clips")
    for index, record in enumerate(result):
        record["index"] = index
    META.mkdir(parents=True, exist_ok=True)
    (META / "samples.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    summaries = {}
    for split in ["train", "test"]:
        selected = [record for record in result if record["split"] == split]
        summaries[split] = {
            "clips": len(selected), "subjects": dict(Counter(record["user"] for record in selected)),
            "classes": dict(Counter(record["label"] for record in selected)),
            "modalities": dict(Counter(name for record in selected for name in record["modalities"])),
        }
    (META / "dataset_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps(summaries, indent=2), flush=True)
    return result


def sensor_clip(record):
    skeleton = np.zeros((64, 153), dtype=np.float32)
    imu = np.zeros((64, 85), dtype=np.float32)
    radar = np.zeros((64, 28), dtype=np.float32)
    present = np.zeros(3, dtype=np.float32)
    paths = {name: ROOT / path for name, path in record["modalities"].items()}
    if "Skeleton" in paths:
        frames = []
        for path in sorted(paths["Skeleton"].rglob("*.json")):
            people = json.loads(path.read_text())
            if isinstance(people, list) and people:
                person = max(people, key=lambda entry: np.mean(entry.get("keypoint_scores", [0])))
                points = np.asarray(person["keypoints"], dtype=np.float32)
                if points.shape != (17, 3):
                    raise ValueError(f"Unexpected skeleton shape {points.shape}: {path}")
                frames.append(points)
        if frames:
            pose = interpolate(np.asarray(frames).reshape(-1, 51)).reshape(64, 17, 3)
            bones = pose - pose[:, PARENTS]
            velocity = np.diff(pose, axis=0, prepend=pose[:1]) * 10
            skeleton = np.concatenate([pose.reshape(64, -1), bones.reshape(64, -1),
                                       velocity.reshape(64, -1)], axis=1)
            present[0] = 1
    if "IMU" in paths:
        tables = [pd.read_csv(path) for path in sorted(paths["IMU"].glob("*.csv"))]
        if tables:
            table = pd.concat(tables, ignore_index=True)
            if not table.empty:
                for device_index, device in enumerate(DEVICES):
                    selected = table[table.iloc[:, 1].astype(str).str.startswith(device + "(")]
                    selected = selected.sort_values(table.columns[0]).drop_duplicates(table.columns[0])
                    if selected.empty:
                        continue
                    numeric = selected.iloc[:, 2:18].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
                    angles = np.deg2rad(numeric[:, 6:9])
                    values = np.concatenate([numeric[:, :3] / 4, numeric[:, 3:6] / 360,
                                             np.sin(angles), np.cos(angles), numeric[:, 12:16]], axis=1)
                    imu[:, device_index * 16:(device_index + 1) * 16] = interpolate(values)
                    imu[:, 80 + device_index] = 1
                present[1] = float(imu[:, 80:].any())
    if "Radar" in paths:
        tables = [pd.read_csv(path) for path in sorted(paths["Radar"].glob("*.csv"))]
        if tables:
            table = pd.concat(tables, ignore_index=True)
            features = []
            for _, frame in table.groupby("timestamp", sort=True):
                values = frame[["x", "y", "z", "v", "snr", "noise"]].to_numpy(dtype=np.float32)
                values = np.nan_to_num(values) / np.asarray([5, 5, 5, 5, 1000, 1000])
                features.append(np.concatenate([values.mean(0), values.std(0), values.min(0),
                                                values.max(0), [np.log1p(len(values)) / 5, 0, 0, 1]]))
            if features:
                radar = interpolate(features)
                present[2] = 1
    return skeleton.clip(-10, 10), imu.clip(-10, 10), radar.clip(-10, 10), present


def prepare_sensors(records, workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        arrays = list(tqdm(pool.map(sensor_clip, records, chunksize=8), total=len(records), desc="Sensors"))
    CACHE.mkdir(parents=True, exist_ok=True)
    for column, name in enumerate(["skeleton", "imu", "radar", "sensor_mask"]):
        values = np.stack([item[column] for item in arrays])
        np.save(CACHE / f"{name}.npy", values)
        print(name, values.shape, "finite", bool(np.isfinite(values).all()), flush=True)


def image_clip(arguments):
    record, steps, size = arguments
    image = np.zeros((steps, 7, size, size), dtype=np.uint8)
    mask = np.zeros(3, dtype=np.float32)
    errors = []
    for modality_index, (modality, start, stop) in enumerate([
        ("Depth_Color", 0, 3), ("IR", 3, 4), ("Thermal", 4, 7),
    ]):
        if modality not in record["modalities"]:
            continue
        files = sorted(path for path in (ROOT / record["modalities"][modality]).iterdir()
                       if path.suffix.lower() in (".png", ".jpg", ".jpeg") and not path.name.startswith("."))
        if not files:
            continue
        indices = np.linspace(0, len(files) - 1, steps).round().astype(int)
        decoded = {}
        for frame_index in np.unique(indices):
            try:
                with Image.open(files[frame_index]) as frame:
                    frame = frame.convert("L" if stop - start == 1 else "RGB")
                    frame = frame.resize((size, size), Image.Resampling.BILINEAR)
                    values = np.asarray(frame)
                    decoded[frame_index] = values[None] if values.ndim == 2 else values.transpose(2, 0, 1)
            except (OSError, ValueError) as error:
                errors.append({"file": str(files[frame_index].relative_to(ROOT)), "error": type(error).__name__})
        if decoded:
            valid = list(decoded)
            for position, frame_index in enumerate(indices):
                closest = min(valid, key=lambda candidate: abs(candidate - frame_index))
                image[position, start:stop] = decoded[closest]
            mask[modality_index] = 1
    destination = CACHE / f"video_{steps}_{size}" / f"{record['index']:05d}.npy"
    np.save(destination, image)
    return mask, errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["index", "sensors", "video"], required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--size", type=int, default=96)
    args = parser.parse_args()
    if args.stage == "index":
        build_index()
        return
    records = json.loads((META / "samples.json").read_text())
    if args.stage == "sensors":
        prepare_sensors(records, args.workers)
    else:
        (CACHE / f"video_{args.frames}_{args.size}").mkdir(parents=True, exist_ok=True)
        tasks = ((record, args.frames, args.size) for record in records)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(tqdm(pool.map(image_clip, tasks, chunksize=4), total=len(records), desc="Video"))
        np.save(CACHE / "video_mask.npy", np.stack([item[0] for item in results]))
        errors = [error for item in results for error in item[1]]
        (META / "image_decode_errors.json").write_text(json.dumps(errors, indent=2) + "\n")
        print("Undecodable selected frames:", len(errors), flush=True)


if __name__ == "__main__":
    main()

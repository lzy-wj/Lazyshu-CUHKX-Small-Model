#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ROOT, get_records


FRAME_PATTERN = re.compile(r"_(\d+)(?:_Color)?\.(?:png|jpg)$", re.IGNORECASE)
MEMMAP = None
WORKER_CONFIG = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/r2plus1d_yolo_16_128")
    parser.add_argument("--yolo", type=Path, default=ROOT / ".tools/checkpoints/yolo11n.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--detection-batch-size", type=int, default=128)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--rgb-modalities", default="Depth_Color")
    parser.add_argument("--windows-from", type=Path)
    return parser.parse_args()


def frame_files(folder):
    folder = Path(folder)
    if not folder.is_dir():
        return []
    files = [path for path in folder.iterdir() if path.suffix.lower() in (".png", ".jpg", ".jpeg")]

    def key(path):
        match = FRAME_PATTERN.search(path.name)
        return (0, int(match.group(1))) if match else (1, path.name)

    return [str(path) for path in sorted(files, key=key)]


def pick_indices(length, count):
    if length == 0:
        return []
    return np.linspace(0, length - 1, count).round().astype(int).tolist()


def readable_detection_path(path):
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError):
        return False


def normalized_window(boxes, width, height, margin=1.4, minimum_side=0.35):
    boxes = np.asarray(boxes, dtype=np.float64)
    left, top = boxes[:, 0].min() / width, boxes[:, 1].min() / height
    right, bottom = boxes[:, 2].max() / width, boxes[:, 3].max() / height
    center_x, center_y = (left + right) / 2, (top + bottom) / 2
    side = max((right - left) * width, (bottom - top) * height) * margin
    side = max(side, minimum_side * max(width, height))
    half_x, half_y = side / width / 2, side / height / 2
    return [
        max(center_x - half_x, 0), max(center_y - half_y, 0),
        min(center_x + half_x, 1), min(center_y + half_y, 1),
    ]


def detect(records, yolo_path, device, batch_size):
    from ultralytics import YOLO

    model = YOLO(str(yolo_path))
    boxes = [[] for _ in records]
    dimensions = [None for _ in records]

    def run(modality, owners):
        paths = []
        path_owners = []
        for owner in owners:
            files = frame_files(records[owner]["modalities"].get(modality, ""))
            for frame_index in sorted(set(pick_indices(len(files), 8))):
                path = files[frame_index]
                if readable_detection_path(path):
                    paths.append(path)
                    path_owners.append(owner)
        if not paths:
            return
        with tqdm(total=len(paths), desc=f"YOLO {modality}") as progress:
            for start in range(0, len(paths), 512):
                chunk_paths = paths[start:start + 512]
                chunk_owners = path_owners[start:start + 512]
                results = model.predict(
                    chunk_paths,
                    classes=[0],
                    conf=0.25,
                    verbose=False,
                    device=device,
                    batch=batch_size,
                )
                for owner, result in zip(chunk_owners, results):
                    height, width = result.orig_shape
                    dimensions[owner] = (width, height)
                    if len(result.boxes):
                        best = int(result.boxes.conf.argmax())
                        boxes[owner].append(result.boxes.xyxy[best].detach().cpu().tolist())
                progress.update(len(chunk_paths))

    run("IR", range(len(records)))
    missing = [index for index, values in enumerate(boxes) if not values]
    run("Depth_Color", missing)
    windows = [
        normalized_window(values, *dimensions[index]) if values and dimensions[index] else None
        for index, values in enumerate(boxes)
    ]
    return windows


def initialize_worker(path, shape, frames, image_size, rgb_modalities):
    global MEMMAP, WORKER_CONFIG
    MEMMAP = np.memmap(path, dtype=np.uint8, mode="r+", shape=shape)
    WORKER_CONFIG = {
        "frames": frames,
        "image_size": image_size,
        "rgb_modalities": rgb_modalities,
    }


def read_frame(path, mode, window):
    try:
        with Image.open(path) as image:
            image = image.convert(mode)
            if window is not None:
                width, height = image.size
                image = image.crop(
                    (
                        round(window[0] * width), round(window[1] * height),
                        round(window[2] * width), round(window[3] * height),
                    )
                )
            array = np.asarray(
                image.resize((WORKER_CONFIG["image_size"], WORKER_CONFIG["image_size"]), Image.Resampling.BILINEAR),
                dtype=np.uint8,
            )
        return None if not array.any() else array
    except (OSError, ValueError):
        return None


def process_record(job):
    index, record, window = job
    frames = WORKER_CONFIG["frames"]
    buffer = np.zeros(MEMMAP.shape[1:], dtype=np.uint8)
    failures = {modality: 0 for modality in WORKER_CONFIG["rgb_modalities"] + ["IR"]}
    channel = 0
    streams = [(modality, "RGB", 3) for modality in WORKER_CONFIG["rgb_modalities"]]
    streams.append(("IR", "L", 1))
    for modality, mode, channels in streams:
        files = frame_files(record["modalities"].get(modality, ""))
        if not files:
            failures[modality] = frames
            channel += channels
            continue
        for output_index, source_index in enumerate(pick_indices(len(files), frames)):
            array = read_frame(files[source_index], mode, window)
            if array is None:
                failures[modality] += 1
            elif channels == 1:
                buffer[output_index, channel] = array
            else:
                buffer[output_index, channel:channel + channels] = array.transpose(2, 0, 1)
        channel += channels
    MEMMAP[index] = buffer
    return index, failures


def save_availability(cache_path, shape, output_path):
    cache = np.memmap(cache_path, dtype=np.uint8, mode="r", shape=shape)
    available = np.zeros(shape[0], dtype=np.uint8)
    for start in range(0, shape[0], 16):
        chunk = np.asarray(cache[start:start + 16])
        available[start:start + len(chunk)] = chunk.reshape(len(chunk), -1).any(axis=1)
    np.save(output_path, available)


def main():
    args = parse_args()
    if args.frames < 8 or args.image_size < 96:
        raise ValueError("Video caches require at least 8 frames and 96 pixels")
    records = get_records()
    rgb_modalities = [value.strip() for value in args.rgb_modalities.split(",") if value.strip()]
    if not rgb_modalities or len(set(rgb_modalities)) != len(rgb_modalities):
        raise ValueError("RGB modalities must be a nonempty unique list")
    if set(rgb_modalities) - {"Depth_Color", "Thermal"}:
        raise ValueError("Only Depth_Color and Thermal RGB streams are supported")
    args.output.mkdir(parents=True, exist_ok=True)
    windows_path = args.output / "windows.json"
    if windows_path.is_file():
        windows = json.loads(windows_path.read_text())
        if len(windows) != len(records):
            raise ValueError("Existing YOLO windows do not match metadata")
    elif args.windows_from is not None:
        windows = json.loads(args.windows_from.read_text())
        if len(windows) != len(records):
            raise ValueError("Reused YOLO windows do not match metadata")
        windows_path.write_text(json.dumps(windows) + "\n")
    else:
        windows = detect(records, args.yolo, args.device, args.detection_batch_size)
        windows_path.write_text(json.dumps(windows) + "\n")

    channels = 3 * len(rgb_modalities) + 1
    shape = (len(records), args.frames, channels, args.image_size, args.image_size)
    cache_path = args.output / "frames.uint8.memmap"
    expected_bytes = int(np.prod(shape))
    manifest_path = args.output / "manifest.json"
    if manifest_path.is_file() and cache_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("complete") and manifest.get("shape") == list(shape) and cache_path.stat().st_size == expected_bytes:
            availability_path = args.output / "available.npy"
            if not availability_path.is_file():
                save_availability(cache_path, shape, availability_path)
            print(f"Complete cache already exists: {cache_path}")
            return
    np.memmap(cache_path, dtype=np.uint8, mode="w+", shape=shape).flush()
    failures = {modality: 0 for modality in rgb_modalities + ["IR"]}
    jobs = [(record["index"], record, windows[record["index"]]) for record in records]
    with Pool(
        args.workers,
        initializer=initialize_worker,
        initargs=(str(cache_path), shape, args.frames, args.image_size, rgb_modalities),
    ) as pool:
        for _, result in tqdm(pool.imap_unordered(process_record, jobs, chunksize=4), total=len(jobs), desc="cache"):
            for key in failures:
                failures[key] += result[key]
    manifest = {
        "complete": True,
        "shape": list(shape),
        "bytes": cache_path.stat().st_size,
        "metadata_sha256": hashlib.sha256((ROOT / "data/metadata/samples.json").read_bytes()).hexdigest(),
        "yolo_sha256": hashlib.sha256(args.yolo.read_bytes()).hexdigest(),
        "detected_clips": sum(window is not None for window in windows),
        "fallback_clips": sum(window is None for window in windows),
        "failed_frames": failures,
        "rgb_modalities": rgb_modalities,
        "modalities": [f"{modality} RGB" for modality in rgb_modalities] + ["IR grayscale"],
        "channels": channels,
        "normalization": "Kinetics RGB statistics per RGB stream; channel-averaged Kinetics statistics for IR",
        "crop": "union of up to eight best person detections, margin 1.4, minimum side 0.35",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    save_availability(cache_path, shape, args.output / "available.npy")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

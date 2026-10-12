#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import get_records


FRAME_PATTERN = re.compile(r"_(\d+)(?:_Color)?\.(?:png|jpg)$", re.IGNORECASE)
MEMMAP = None
WORKER_CONFIG = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=EXPERIMENT / "cache/ablation_dynamic_crop_32_128")
    parser.add_argument("--yolo", type=Path, default=ROOT / ".tools/checkpoints/yolo11n.pt")
    parser.add_argument("--fixed-windows", type=Path,
                        default=ROOT / "data/processed/r2plus1d_yolo_32_128/windows.json")
    parser.add_argument("--device", default="7")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--detection-batch-size", type=int, default=128)
    parser.add_argument("--detection-chunk-size", type=int, default=512)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--margin", type=float, default=1.4)
    parser.add_argument("--minimum-side", type=float, default=0.35)
    parser.add_argument("--stage", choices=("detect", "cache", "all"), default="all")
    return parser.parse_args()


def sha256(path, chunk_size=16 * 1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


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


def sampled_paths(record, modality, frames):
    files = frame_files(record["modalities"].get(modality, ""))
    return [files[index] for index in pick_indices(len(files), frames)] if files else [None] * frames


def readable(path):
    if path is None:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError):
        return False


def box_iou(left, right):
    intersection_left = max(left[0], right[0])
    intersection_top = max(left[1], right[1])
    intersection_right = min(left[2], right[2])
    intersection_bottom = min(left[3], right[3])
    intersection = max(0.0, intersection_right - intersection_left) * max(
        0.0, intersection_bottom - intersection_top
    )
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(left_area + right_area - intersection, 1e-12)


def center_similarity(left, right):
    left_center = ((left[0] + left[2]) / 2, (left[1] + left[3]) / 2)
    right_center = ((right[0] + right[2]) / 2, (right[1] + right[3]) / 2)
    distance = np.hypot(left_center[0] - right_center[0], left_center[1] - right_center[1])
    return float(np.exp(-4 * distance))


def associate(candidates):
    selected = [None] * len(candidates)
    previous = None
    previous_time = None
    for time, values in enumerate(candidates):
        if not values:
            continue
        if previous is None:
            choice = max(values, key=lambda value: value[4])
        else:
            gap = max(time - previous_time, 1)
            continuity = np.exp(-0.12 * (gap - 1))
            choice = max(
                values,
                key=lambda value: (
                    0.55 * value[4]
                    + continuity * (0.30 * box_iou(previous, value) + 0.15 * center_similarity(previous, value))
                ),
            )
        selected[time] = choice
        previous = choice
        previous_time = time
    return selected


def geometry(box, margin, minimum_side):
    left, top, right, bottom, _, width, height = box
    maximum = max(width, height)
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    side = max((right - left) * width, (bottom - top) * height) * margin / maximum
    return np.asarray([center_x, center_y, np.log(max(side, minimum_side))], dtype=np.float64)


def fixed_geometry(window, width=640, height=480):
    if window is None:
        window = [0, 0, 1, 1]
    left, top, right, bottom = window
    maximum = max(width, height)
    side = max((right - left) * width, (bottom - top) * height) / maximum
    return np.asarray([(left + right) / 2, (top + bottom) / 2, np.log(max(side, 1e-6))])


def smooth_geometry(values):
    kernel = np.asarray([1, 2, 3, 2, 1], dtype=np.float64)
    kernel /= kernel.sum()
    padded = np.pad(values, ((2, 2), (0, 0)), mode="edge")
    return np.stack([
        np.convolve(padded[:, column], kernel, mode="valid") for column in range(values.shape[1])
    ], axis=1)


def geometry_window(value, width=640, height=480):
    center_x, center_y, log_side = value
    maximum = max(width, height)
    side = np.exp(log_side) * maximum
    half_x = side / width / 2
    half_y = side / height / 2
    return np.asarray([
        max(center_x - half_x, 0), max(center_y - half_y, 0),
        min(center_x + half_x, 1), min(center_y + half_y, 1),
    ], dtype=np.float32)


def detect_modality(model, records, candidates, dimensions, sources, modality, source_id, args,
                    only_missing):
    path_owners = defaultdict(list)
    for record in records:
        owner = record["index"]
        paths = sampled_paths(record, modality, args.frames)
        for time, path in enumerate(paths):
            if only_missing and candidates[owner][time]:
                continue
            if readable(path):
                path_owners[path].append((owner, time))
    paths = list(path_owners)
    print(json.dumps({"modality": modality, "unique_images": len(paths), "positions": sum(
        len(values) for values in path_owners.values()
    )}), flush=True)
    with tqdm(total=len(paths), desc=f"YOLO {modality}") as progress:
        for start in range(0, len(paths), args.detection_chunk_size):
            chunk = paths[start:start + args.detection_chunk_size]
            results = model.predict(
                chunk,
                classes=[0],
                conf=0.25,
                verbose=False,
                device=args.device,
                batch=args.detection_batch_size,
            )
            if len(results) != len(chunk):
                raise RuntimeError("YOLO result count does not match input count")
            for path, result in zip(chunk, results):
                height, width = result.orig_shape
                values = []
                if len(result.boxes):
                    boxes = result.boxes.xyxy.detach().cpu().numpy()
                    confidence = result.boxes.conf.detach().cpu().numpy()
                    for box, score in zip(boxes, confidence):
                        values.append([
                            float(box[0] / width), float(box[1] / height),
                            float(box[2] / width), float(box[3] / height),
                            float(score), float(width), float(height),
                        ])
                for owner, time in path_owners[path]:
                    if values:
                        candidates[owner][time] = values
                        dimensions[owner, time] = (width, height)
                        sources[owner, time] = source_id
            progress.update(len(chunk))


def detection_stage(records, fixed_windows, args):
    from ultralytics import YOLO

    model = YOLO(str(args.yolo))
    candidates = [[[] for _ in range(args.frames)] for _ in records]
    dimensions = np.zeros((len(records), args.frames, 2), dtype=np.int16)
    sources = np.zeros((len(records), args.frames), dtype=np.uint8)
    detect_modality(model, records, candidates, dimensions, sources, "IR", 1, args, only_missing=False)
    detect_modality(model, records, candidates, dimensions, sources, "Depth_Color", 2, args, only_missing=True)

    selected_boxes = np.full((len(records), args.frames, 7), np.nan, dtype=np.float32)
    interpolated_geometry = np.empty((len(records), args.frames, 3), dtype=np.float32)
    smoothed_geometry = np.empty_like(interpolated_geometry)
    windows = np.empty((len(records), args.frames, 4), dtype=np.float32)
    selected_confidence = np.full((len(records), args.frames), np.nan, dtype=np.float32)
    for record in tqdm(records, desc="tracking"):
        owner = record["index"]
        selected = associate(candidates[owner])
        available_times = [time for time, box in enumerate(selected) if box is not None]
        values = np.full((args.frames, 3), np.nan, dtype=np.float64)
        for time in available_times:
            selected_boxes[owner, time] = selected[time]
            selected_confidence[owner, time] = selected[time][4]
            values[time] = geometry(selected[time], args.margin, args.minimum_side)
        if available_times:
            times = np.arange(args.frames)
            for column in range(3):
                values[:, column] = np.interp(
                    times, available_times, values[available_times, column]
                )
        else:
            values[:] = fixed_geometry(fixed_windows[owner])
        smooth = smooth_geometry(values)
        interpolated_geometry[owner] = values
        smoothed_geometry[owner] = smooth
        fallback_dimensions = next(
            (dimensions[owner, time] for time in available_times if dimensions[owner, time].all()),
            np.asarray([640, 480]),
        )
        for time in range(args.frames):
            width, height = dimensions[owner, time]
            if not width or not height:
                width, height = fallback_dimensions
            windows[owner, time] = geometry_window(smooth[time], int(width), int(height))

    tracking_path = args.output / "tracking.npz"
    np.savez_compressed(
        tracking_path,
        windows=windows,
        selected_boxes=selected_boxes,
        selected_confidence=selected_confidence,
        detection_source=sources,
        dimensions=dimensions,
        interpolated_geometry=interpolated_geometry,
        smoothed_geometry=smoothed_geometry,
    )
    audit = audit_tracking(
        records, fixed_windows, windows, selected_boxes, selected_confidence, sources,
        interpolated_geometry, smoothed_geometry, args,
    )
    audit["tracking_sha256"] = sha256(tracking_path)
    atomic_json(args.output / "box_audit.json", audit)
    print(json.dumps(audit, indent=2), flush=True)


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "p99": float(np.quantile(values, 0.99)),
    }


def audit_tracking(records, fixed_windows, windows, selected_boxes, confidence, sources,
                   interpolated, smoothed, args):
    fixed_iou = []
    center_shift = []
    area_ratio = []
    contained = []
    for record in records:
        owner = record["index"]
        fixed = fixed_windows[owner] if fixed_windows[owner] is not None else [0, 0, 1, 1]
        fixed_area = max((fixed[2] - fixed[0]) * (fixed[3] - fixed[1]), 1e-12)
        fixed_center = np.asarray([(fixed[0] + fixed[2]) / 2, (fixed[1] + fixed[3]) / 2])
        for time, window in enumerate(windows[owner]):
            fixed_iou.append(box_iou(fixed, window))
            center = np.asarray([(window[0] + window[2]) / 2, (window[1] + window[3]) / 2])
            center_shift.append(float(np.linalg.norm(center - fixed_center)))
            area_ratio.append(float((window[2] - window[0]) * (window[3] - window[1]) / fixed_area))
            box = selected_boxes[owner, time]
            if np.isfinite(box).all():
                contained.append(float(
                    box[0] >= window[0] and box[1] >= window[1]
                    and box[2] <= window[2] and box[3] <= window[3]
                ))
    raw_delta = np.diff(interpolated[:, :, :2], axis=1)
    smooth_delta = np.diff(smoothed[:, :, :2], axis=1)
    raw_side_delta = np.abs(np.diff(interpolated[:, :, 2], axis=1))
    smooth_side_delta = np.abs(np.diff(smoothed[:, :, 2], axis=1))
    detected = sources > 0
    return {
        "records": len(records),
        "frames_per_record": args.frames,
        "positions": int(detected.size),
        "detected_positions": int(detected.sum()),
        "detection_coverage": float(detected.mean()),
        "ir_detection_coverage": float((sources == 1).mean()),
        "rgb_fallback_coverage": float((sources == 2).mean()),
        "clips_with_detection": int(detected.any(axis=1).sum()),
        "clips_without_detection": int((~detected.any(axis=1)).sum()),
        "mean_selected_confidence": float(np.nanmean(confidence)),
        "center_jitter_before_smoothing": summarize(np.linalg.norm(raw_delta, axis=2).ravel()),
        "center_jitter_after_smoothing": summarize(np.linalg.norm(smooth_delta, axis=2).ravel()),
        "log_side_jitter_before_smoothing": summarize(raw_side_delta.ravel()),
        "log_side_jitter_after_smoothing": summarize(smooth_side_delta.ravel()),
        "dynamic_fixed_window_iou": summarize(fixed_iou),
        "dynamic_fixed_center_shift": summarize(center_shift),
        "dynamic_to_fixed_area_ratio": summarize(area_ratio),
        "selected_detection_fully_inside_smoothed_crop": float(np.mean(contained)),
        "association": "greedy confidence + IoU + center continuity across observed frames",
        "missing_boxes": "linear interpolation; clip-level fixed-window fallback only when no frame is detected",
        "smoothing": "five-frame triangular smoothing [1,2,3,2,1] of center and log side",
        "crop": {
            "margin": args.margin,
            "minimum_side_relative_to_long_image_edge": args.minimum_side,
        },
    }


def initialize_worker(path, shape, frames, image_size):
    global MEMMAP, WORKER_CONFIG
    MEMMAP = np.memmap(path, dtype=np.uint8, mode="r+", shape=shape)
    WORKER_CONFIG = {"frames": frames, "image_size": image_size}


def read_frame(path, mode, window):
    try:
        with Image.open(path) as image:
            image = image.convert(mode)
            width, height = image.size
            image = image.crop((
                round(window[0] * width), round(window[1] * height),
                round(window[2] * width), round(window[3] * height),
            ))
            array = np.asarray(
                image.resize(
                    (WORKER_CONFIG["image_size"], WORKER_CONFIG["image_size"]),
                    Image.Resampling.BILINEAR,
                ),
                dtype=np.uint8,
            )
        return None if not array.any() else array
    except (OSError, ValueError):
        return None


def process_record(job):
    owner, record, windows = job
    frames = WORKER_CONFIG["frames"]
    buffer = np.zeros(MEMMAP.shape[1:], dtype=np.uint8)
    failures = {"Depth_Color": 0, "IR": 0}
    streams = (("Depth_Color", "RGB", 0), ("IR", "L", 3))
    for modality, mode, channel in streams:
        paths = sampled_paths(record, modality, frames)
        for time, path in enumerate(paths):
            if path is None:
                failures[modality] += 1
                continue
            array = read_frame(path, mode, windows[time])
            if array is None:
                failures[modality] += 1
            elif mode == "L":
                buffer[time, channel] = array
            else:
                buffer[time, channel:channel + 3] = array.transpose(2, 0, 1)
    MEMMAP[owner] = buffer
    return failures


def save_availability(cache_path, shape, output_path):
    cache = np.memmap(cache_path, dtype=np.uint8, mode="r", shape=shape)
    available = np.zeros(shape[0], dtype=np.uint8)
    for start in range(0, shape[0], 16):
        chunk = np.asarray(cache[start:start + 16])
        available[start:start + len(chunk)] = chunk.reshape(len(chunk), -1).any(axis=1)
    np.save(output_path, available)


def cache_stage(records, args):
    tracking_path = args.output / "tracking.npz"
    if not tracking_path.is_file():
        raise FileNotFoundError("Run the detection stage before cache generation")
    tracking = np.load(tracking_path)
    windows = tracking["windows"]
    expected_windows_shape = (len(records), args.frames, 4)
    if windows.shape != expected_windows_shape or not np.isfinite(windows).all():
        raise ValueError("Tracking windows are incomplete")
    shape = (len(records), args.frames, 4, args.image_size, args.image_size)
    cache_path = args.output / "frames.uint8.memmap"
    manifest_path = args.output / "manifest.json"
    expected_bytes = int(np.prod(shape))
    if manifest_path.is_file() and cache_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("complete") and manifest.get("shape") == list(shape) \
                and cache_path.stat().st_size == expected_bytes:
            print(f"Complete cache already exists: {cache_path}")
            return
    atomic_json(manifest_path, {"complete": False, "shape": list(shape), "expected_bytes": expected_bytes})
    np.memmap(cache_path, dtype=np.uint8, mode="w+", shape=shape).flush()
    failures = {"Depth_Color": 0, "IR": 0}
    jobs = [(record["index"], record, windows[record["index"]]) for record in records]
    with Pool(
        args.workers,
        initializer=initialize_worker,
        initargs=(str(cache_path), shape, args.frames, args.image_size),
    ) as pool:
        for result in tqdm(pool.imap_unordered(process_record, jobs, chunksize=4),
                           total=len(jobs), desc="cache"):
            for modality in failures:
                failures[modality] += result[modality]
    save_availability(cache_path, shape, args.output / "available.npy")
    cache_digest = sha256(cache_path)
    manifest = {
        "complete": True,
        "experiment": "ablation_dynamic_crop",
        "shape": list(shape),
        "bytes": cache_path.stat().st_size,
        "sha256": cache_digest,
        "metadata_sha256": sha256(ROOT / "data/metadata/samples.json"),
        "yolo_sha256": sha256(args.yolo),
        "fixed_windows_sha256": sha256(args.fixed_windows),
        "tracking_sha256": sha256(tracking_path),
        "failed_frames": failures,
        "rgb_modalities": ["Depth_Color"],
        "modalities": ["Depth_Color RGB", "IR grayscale"],
        "channels": 4,
        "frames": args.frames,
        "image_size": args.image_size,
        "normalization": "Kinetics RGB statistics; channel-averaged Kinetics statistics for IR",
        "sample_positions": "identical np.linspace(...).round() positions as frozen cache",
        "crop": "per-frame YOLO person track, missing interpolation, five-frame triangular smoothing",
    }
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2), flush=True)


def main():
    args = parse_args()
    if args.frames != 32 or args.image_size != 128:
        raise ValueError("This ablation is frozen to 32 frames and 128 pixels")
    if not args.output.resolve().is_relative_to(EXPERIMENT.resolve()):
        raise ValueError("Dynamic-crop cache must stay inside experiments/dynamic_crop")
    args.output.mkdir(parents=True, exist_ok=True)
    records = get_records()
    if [record["index"] for record in records] != list(range(len(records))):
        raise ValueError("Record indices must be contiguous")
    fixed_windows = json.loads(args.fixed_windows.read_text())
    if len(fixed_windows) != len(records):
        raise ValueError("Fixed windows do not match metadata")
    if args.stage in ("detect", "all"):
        detection_stage(records, fixed_windows, args)
    if args.stage in ("cache", "all"):
        cache_stage(records, args)


if __name__ == "__main__":
    main()

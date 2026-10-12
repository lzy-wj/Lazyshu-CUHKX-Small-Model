#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.prepare_r2plus1d_cache import (
    frame_files,
    pick_indices,
    readable_detection_path,
)
from src.data import get_records


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=("Depth_Color", "IR", "Thermal"), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--yolo", type=Path, default=ROOT / ".tools/checkpoints/yolo11n.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def robust_window(detections, sampled_frames):
    if not detections:
        return None, 0.0
    boxes = np.asarray([value[:4] for value in detections], dtype=np.float64)
    confidence = np.asarray([value[4] for value in detections], dtype=np.float64)
    centers = np.stack(
        [(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1
    )
    median = np.median(centers, axis=0)
    distance = np.linalg.norm(centers - median, axis=1)
    threshold = max(0.12, float(np.median(distance) * 2.5 + 0.03))
    inliers = distance <= threshold
    if not inliers.any():
        inliers[confidence.argmax()] = True
    boxes = boxes[inliers]
    confidence = confidence[inliers]
    left, top = boxes[:, :2].min(axis=0)
    right, bottom = boxes[:, 2:].max(axis=0)
    center_x, center_y = (left + right) / 2, (top + bottom) / 2
    side = max(right - left, bottom - top) * 1.45
    side = max(side, 0.35)
    window = [
        max(center_x - side / 2, 0.0),
        max(center_y - side / 2, 0.0),
        min(center_x + side / 2, 1.0),
        min(center_y + side / 2, 1.0),
    ]
    coverage = len(confidence) / max(sampled_frames, 1)
    consistency = max(0.25, 1.0 - float(np.median(distance[inliers])) * 2)
    quality = coverage * float(np.median(confidence)) * consistency
    return window, quality


def main():
    args = parse_args()
    from ultralytics import YOLO

    records = get_records()
    paths = []
    owners = []
    sampled = np.zeros(len(records), dtype=np.int64)
    for record in records:
        files = frame_files(record["modalities"].get(args.modality, ""))
        positions = sorted(set(pick_indices(len(files), args.frames)))
        sampled[record["index"]] = len(positions)
        for position in positions:
            if readable_detection_path(files[position]):
                paths.append(files[position])
                owners.append(record["index"])
    model = YOLO(str(args.yolo))
    detections = [[] for _ in records]
    with tqdm(total=len(paths), desc=args.modality) as progress:
        for start in range(0, len(paths), 512):
            chunk = paths[start : start + 512]
            results = model.predict(
                chunk,
                classes=[0],
                conf=0.20,
                verbose=False,
                device=args.device,
                batch=args.batch_size,
            )
            for owner, result in zip(owners[start : start + 512], results):
                if not len(result.boxes):
                    continue
                best = int(result.boxes.conf.argmax())
                height, width = result.orig_shape
                box = result.boxes.xyxy[best].detach().cpu().numpy()
                detections[owner].append(
                    [
                        float(box[0] / width),
                        float(box[1] / height),
                        float(box[2] / width),
                        float(box[3] / height),
                        float(result.boxes.conf[best]),
                    ]
                )
            progress.update(len(chunk))
    windows = []
    quality = []
    for index, values in enumerate(detections):
        window, score = robust_window(values, int(sampled[index]))
        windows.append(window)
        quality.append(score)
    report = {
        "modality": args.modality,
        "windows": windows,
        "quality": quality,
        "detections": [len(values) for values in detections],
        "sampled_frames": sampled.tolist(),
        "covered_clips": int(sum(window is not None for window in windows)),
        "mean_quality": float(np.mean(quality)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report) + "\n")
    print(json.dumps({key: value for key, value in report.items() if not isinstance(value, list)}, indent=2))


if __name__ == "__main__":
    main()

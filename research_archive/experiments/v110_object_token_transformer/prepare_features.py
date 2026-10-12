#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "data/processed/r2plus1d_yolo_all_visual_32_192",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=ROOT / "outputs/final_legal_multimodal_v9/models/yolo11n_compact.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/processed/yolo_object_tokens_8x3",
    )
    parser.add_argument("--device", required=True)
    parser.add_argument("--clip-batch-size", type=int, default=24)
    parser.add_argument("--inference-batch-size", type=int, default=128)
    parser.add_argument("--confidence", type=float, default=0.05)
    args = parser.parse_args()
    if int(args.device) not in range(4, 8):
        raise ValueError("V110 extraction may only use physical GPUs 4-7")
    used = int(
        __import__("subprocess")
        .check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "-i",
                args.device,
            ],
            text=True,
        )
        .strip()
    )
    if used > 1024:
        raise RuntimeError(f"GPU {args.device} is busy ({used} MiB)")

    cache_manifest_path = args.cache_dir / "manifest.json"
    cache_manifest = json.loads(cache_manifest_path.read_text())
    shape = tuple(cache_manifest["shape"])
    if not cache_manifest.get("complete") or shape[1:] != (32, 7, 192, 192):
        raise ValueError("Unexpected V110 visual cache")
    args.output.mkdir(parents=True, exist_ok=True)
    feature_path = args.output / "features.float16.npy"
    manifest_path = args.output / "manifest.json"
    expected_shape = (shape[0], 8, 3, 80, 6)
    if manifest_path.is_file() and feature_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("complete") and manifest.get("shape") == list(expected_shape):
            print(json.dumps(manifest, indent=2))
            return

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    cache = np.memmap(
        args.cache_dir / "frames.uint8.memmap",
        dtype=np.uint8,
        mode="r",
        shape=shape,
    )
    features = np.zeros(expected_shape, dtype=np.float16)
    phases = np.linspace(0, 31, 8).round().astype(np.int64)
    detection_count = 0
    observation_count = 0
    progress = tqdm(total=shape[0], desc="YOLO object tokens")
    for clip_start in range(0, shape[0], args.clip_batch_size):
        clip_stop = min(clip_start + args.clip_batch_size, shape[0])
        images = []
        owners = []
        for clip in range(clip_start, clip_stop):
            for phase_position, frame in enumerate(phases):
                values = cache[clip, frame]
                streams = (
                    values[:3].transpose(1, 2, 0),
                    values[3:6].transpose(1, 2, 0),
                    np.repeat(values[6:,].transpose(1, 2, 0), 3, axis=2),
                )
                for modality, image in enumerate(streams):
                    images.append(np.asarray(image).copy())
                    owners.append((clip, phase_position, modality))
        results = model.predict(
            images,
            conf=args.confidence,
            iou=0.5,
            max_det=20,
            imgsz=256,
            batch=args.inference_batch_size,
            device=args.device,
            half=True,
            verbose=False,
        )
        for owner, result in zip(owners, results):
            clip, phase_position, modality = owner
            boxes = result.boxes
            if len(boxes):
                classes = boxes.cls.detach().cpu().numpy().astype(np.int64)
                confidence = boxes.conf.detach().cpu().numpy()
                xywh = boxes.xywhn.detach().cpu().numpy()
                for class_index in np.unique(classes):
                    selected = np.flatnonzero(classes == class_index)
                    best = selected[confidence[selected].argmax()]
                    count = min(len(selected), 10) / 10
                    values = np.asarray(
                        [confidence[best], *xywh[best], count], dtype=np.float32
                    )
                    features[clip, phase_position, modality, class_index] = values
                detection_count += len(boxes)
                observation_count += 1
        progress.update(clip_stop - clip_start)
    progress.close()
    np.save(feature_path, features)
    manifest = {
        "complete": True,
        "shape": list(expected_shape),
        "dtype": "float16",
        "phases": phases.tolist(),
        "modalities": ["Depth_Color", "Thermal", "IR"],
        "classes": 80,
        "features": ["confidence", "center_x", "center_y", "width", "height", "count_div_10"],
        "confidence_threshold": args.confidence,
        "detections": detection_count,
        "nonempty_observations": observation_count,
        "total_observations": shape[0] * 8 * 3,
        "weights": str(args.weights),
        "weights_sha256": sha256(args.weights),
        "cache_manifest_sha256": sha256(cache_manifest_path),
        "script_sha256": sha256(__file__),
        "protocol": "label-independent pretrained COCO detection over fixed temporal phases and all visual streams",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

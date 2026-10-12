#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "data/processed/r2plus1d_yolo_all_visual_32_192",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/processed/r2plus1d_center_detail_32_128",
    )
    parser.add_argument("--chunk-size", type=int, default=8)
    args = parser.parse_args()

    source_manifest_path = args.source / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    source_shape = tuple(source_manifest["shape"])
    if not source_manifest.get("complete") or source_shape[3:] != (192, 192):
        raise ValueError("V108 requires a complete 192-pixel source cache")
    if args.chunk_size <= 0:
        raise ValueError("Chunk size must be positive")

    output_shape = source_shape[:3] + (128, 128)
    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / "frames.uint8.memmap"
    manifest_path = args.output / "manifest.json"
    expected_bytes = int(np.prod(output_shape))
    if manifest_path.is_file() and output_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("complete")
            and manifest.get("shape") == list(output_shape)
            and output_path.stat().st_size == expected_bytes
        ):
            print(json.dumps(manifest, indent=2))
            return

    source = np.memmap(
        args.source / "frames.uint8.memmap",
        dtype=np.uint8,
        mode="r",
        shape=source_shape,
    )
    output = np.memmap(
        output_path,
        dtype=np.uint8,
        mode="w+",
        shape=output_shape,
    )
    top = left = (192 - 128) // 2
    for start in tqdm(
        range(0, source_shape[0], args.chunk_size), desc="center-detail cache"
    ):
        stop = min(start + args.chunk_size, source_shape[0])
        output[start:stop] = source[start:stop, :, :, top : top + 128, left : left + 128]
    output.flush()
    del output

    availability = np.zeros(source_shape[0], dtype=np.uint8)
    completed = np.memmap(
        output_path,
        dtype=np.uint8,
        mode="r",
        shape=output_shape,
    )
    for start in range(0, source_shape[0], 16):
        chunk = np.asarray(completed[start : start + 16])
        availability[start : start + len(chunk)] = chunk.reshape(len(chunk), -1).any(1)
    np.save(args.output / "available.npy", availability)

    manifest = {
        **source_manifest,
        "shape": list(output_shape),
        "bytes": output_path.stat().st_size,
        "complete": True,
        "crop": "center 128x128 pixels from label-independent 192x192 YOLO person window",
        "crop_bounds_in_source_pixels": [left, top, left + 128, top + 128],
        "effective_detail_scale": 1.5,
        "interpolation": "none",
        "source_cache": str(args.source),
        "source_manifest_sha256": sha256(source_manifest_path),
        "available_clips": int(availability.sum()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

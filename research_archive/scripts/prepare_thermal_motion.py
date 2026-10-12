import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ROOT, get_records


def motion_box(video):
    difference = np.diff(video.astype(np.float32), axis=0)
    difference -= np.median(difference, axis=(-2, -1), keepdims=True)
    energy = np.abs(difference).mean((0, 1))
    smooth = np.asarray(Image.fromarray(energy).convert("L").filter(ImageFilter.GaussianBlur(3)), dtype=np.float32)
    floor = np.median(smooth)
    weight = np.maximum(smooth - floor, 0) ** 2
    if weight.sum() < 1 or np.percentile(smooth, 99) < 2:
        return [0.0, 0.0, 1.0, 1.0]
    bounds = []
    for mass in [weight.sum(0), weight.sum(1)]:
        cumulative = np.cumsum(mass) / mass.sum()
        lower, upper = np.searchsorted(cumulative, [0.02, 0.98]) / len(mass)
        center = (lower + upper) / 2
        extent = min(1.0, max(0.45, (upper - lower) * 1.3))
        lower = np.clip(center - extent / 2, 0, 1 - extent)
        bounds.append((float(lower), float(lower + extent)))
    return [bounds[0][0], bounds[1][0], bounds[0][1], bounds[1][1]]


def prepare(record):
    index = record["index"]
    destination = ROOT / "data/processed/thermal_motion_24_160" / f"{index:05d}.npy"
    original = np.load(ROOT / f"data/processed/video_24_160/{index:05d}.npy", mmap_mode="r")[:, 4:]
    box = motion_box(original)
    output = np.array(original)
    errors = []
    if "Thermal" in record["modalities"]:
        files = sorted(path for path in (ROOT / record["modalities"]["Thermal"]).iterdir()
                       if path.suffix.lower() in [".png", ".jpg", ".jpeg"] and not path.name.startswith("."))
        if files:
            indices = np.linspace(0, len(files) - 1, 24).round().astype(int)
            decoded = {}
            for frame_index in np.unique(indices):
                try:
                    with Image.open(files[frame_index]) as frame:
                        frame = frame.convert("RGB")
                        width, height = frame.size
                        pixels = (int(box[0] * width), int(box[1] * height),
                                  max(int(box[2] * width), int(box[0] * width) + 1),
                                  max(int(box[3] * height), int(box[1] * height) + 1))
                        frame = frame.crop(pixels).resize((160, 160), Image.Resampling.BILINEAR)
                        decoded[frame_index] = np.asarray(frame).transpose(2, 0, 1)
                except (OSError, ValueError) as error:
                    errors.append({"file": str(files[frame_index].relative_to(ROOT)), "error": type(error).__name__})
            if decoded:
                for position, frame_index in enumerate(indices):
                    closest = min(decoded, key=lambda candidate: abs(candidate - frame_index))
                    output[position] = decoded[closest]
    temporary = destination.with_suffix(".tmp.npy")
    np.save(temporary, output)
    temporary.replace(destination)
    return {"index": index, "box": box, "errors": errors}


def main():
    destination = ROOT / "data/processed/thermal_motion_24_160"
    destination.mkdir(parents=True, exist_ok=True)
    records = get_records()
    with ProcessPoolExecutor(max_workers=8) as pool:
        results = []
        for result in pool.map(prepare, records, chunksize=8):
            results.append(result)
            if len(results) % 250 == 0:
                print("Prepared", len(results), "/", len(records), flush=True)
    boxes = np.asarray([result["box"] for result in results])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    report = {"status": "complete", "clips": len(results), "shape": [24, 3, 160, 160],
              "policy": "Per-clip temporal difference energy, robust mass bounds with 30% context and minimum 45% side; crop raw thermal frames before resize. No learned detector, no labels, no corpus-fitted statistics.",
              "area_quantiles": np.percentile(areas, [0, 25, 50, 75, 100]).tolist(),
              "metadata_sha256": hashlib.sha256((ROOT / "data/metadata/samples.json").read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "clips_detail": results}
    (ROOT / "data/metadata/thermal_motion.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Complete; crop area quantiles", report["area_quantiles"], flush=True)


if __name__ == "__main__":
    main()

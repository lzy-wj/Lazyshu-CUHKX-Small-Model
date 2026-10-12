import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_person_crops import enhance_infrared, image_files
from src.data import ROOT, get_records


def detail_box(box, confidence):
    result = np.asarray(box, dtype=np.float32).copy()
    if confidence > 0:
        result[3] = result[1] + (result[3] - result[1]) * 0.72
    return result


def crop_clip(task):
    record, boxes, confidence, size = task
    images = np.zeros((16, 6, size, size), dtype=np.uint8)
    mask = np.zeros(2, dtype=np.float32)
    failures = []
    for modality_index, modality in enumerate(["IR", "Thermal"]):
        files = image_files(record, modality)
        if not files:
            continue
        box = detail_box(boxes[modality_index], confidence[modality_index])
        selected = np.linspace(0, len(files) - 1, 16).round().astype(int)
        valid = {}
        for frame_index in np.unique(selected):
            try:
                with Image.open(files[frame_index]) as source:
                    image = enhance_infrared(source) if modality == "IR" else source.convert("RGB")
                    width, height = image.size
                    image = image.crop((int(box[0] * width), int(box[1] * height),
                                        max(int(box[0] * width) + 1, int(box[2] * width)),
                                        max(int(box[1] * height) + 1, int(box[3] * height))))
                    valid[frame_index] = np.asarray(image.resize((size, size), Image.Resampling.BILINEAR)).transpose(2, 0, 1)
            except (OSError, ValueError) as error:
                failures.append({"index": record["index"], "modality": modality,
                                 "frame": int(frame_index), "error": str(error)})
        if valid:
            mask[modality_index] = 1
            for position, frame_index in enumerate(selected):
                nearest = min(valid, key=lambda candidate: abs(candidate - frame_index))
                images[position, modality_index * 3:(modality_index + 1) * 3] = valid[nearest]
    destination = ROOT / f"data/processed/person_detail_16_{size}/{record['index']:05d}.npy"
    with destination.with_suffix(".tmp").open("wb") as handle:
        np.save(handle, images)
    destination.with_suffix(".tmp").replace(destination)
    return mask, failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=160)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    records = get_records()
    box_path = ROOT / "data/metadata/person_boxes.npz"
    source = np.load(box_path)
    destination = ROOT / f"data/processed/person_detail_16_{args.size}"
    destination.mkdir(parents=True, exist_ok=True)
    tasks = ((record, source["boxes"][record["index"]], source["confidence"][record["index"]], args.size)
             for record in records)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        outputs = list(tqdm(pool.map(crop_clip, tasks, chunksize=4), total=len(records), desc="Upper-body detail"))
    masks = np.stack([output[0] for output in outputs])
    np.save(ROOT / "data/processed/person_detail_mask.npy", masks)
    report = {"clips": len(records), "size": args.size, "frames": 16, "available": masks.sum(0).tolist(),
              "modalities": ["IR", "Thermal"], "upper_body_fraction": 0.72,
              "fallback": "Keep full detection box when detector confidence is zero.",
              "protocol": "Fixed label-independent crop of original frames, same transform for all splits.",
              "box_sha256": hashlib.sha256(box_path.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "errors": [error for output in outputs for error in output[1]]}
    (ROOT / "data/metadata/person_detail.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "errors"}), flush=True)


if __name__ == "__main__":
    main()

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.extract_visual_features import enhance_infrared
from src.data import ROOT, get_records
from src.r2plus1d_models import dequantize_state_dict, expand_compact_quantized_state_dict


def image_files(record, modality):
    return sorted(path for path in (ROOT / record["modalities"].get(modality, "missing")).glob("*")
                  if path.suffix.lower() in [".png", ".jpg"] and not path.name.startswith("."))


class DetectionImages(Dataset):
    def __init__(self):
        self.records = get_records()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        images = []
        for modality in ["IR", "Thermal"]:
            files = image_files(self.records[index], modality)
            for fraction in [0.15, 0.5, 0.85]:
                image = Image.new("RGB", (640, 480))
                if files:
                    try:
                        with Image.open(files[round((len(files) - 1) * fraction)]) as source:
                            image = enhance_infrared(source) if modality == "IR" else source.convert("RGB")
                    except (OSError, ValueError):
                        pass
                images.append(np.asarray(image.resize((320, 240))).transpose(2, 0, 1).copy())
        return np.stack(images), index


def detect(weights_path=None):
    os.environ.setdefault("TORCH_HOME", str(ROOT / ".tools/torch"))
    from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights, ssdlite320_mobilenet_v3_large
    torch.set_num_threads(6)
    if weights_path is None:
        model = ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1)
    else:
        model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=91)
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        state = (dequantize_state_dict(checkpoint["model_int8"])
                 if "model_int8" in checkpoint else
                 expand_compact_quantized_state_dict(checkpoint["state_dict_compact"]))
        model.load_state_dict(state, strict=True)
    model = model.cuda().eval()
    loader = DataLoader(DetectionImages(), batch_size=8, num_workers=10, pin_memory=True)
    boxes = np.zeros((len(loader.dataset), 2, 4), dtype=np.float32)
    confidence = np.zeros((len(loader.dataset), 2), dtype=np.float32)
    with torch.inference_mode():
        for images, indices in tqdm(loader, desc="Person detection"):
            images = images.flatten(0, 1).cuda(non_blocking=True).float() / 255
            predictions = model(list(images))
            for position, index in enumerate(indices.tolist()):
                for modality in range(2):
                    candidates = []
                    scores = []
                    for frame in range(3):
                        prediction = predictions[position * 6 + modality * 3 + frame]
                        person = (prediction["labels"] == 1) & (prediction["scores"] > 0.2)
                        if person.any():
                            selected = torch.where(person)[0][prediction["scores"][person].argmax()]
                            candidates.append(prediction["boxes"][selected].cpu().numpy() / [320, 240, 320, 240])
                            scores.append(float(prediction["scores"][selected]))
                    if candidates:
                        candidates = np.stack(candidates)
                        box = np.array([candidates[:, 0].min(), candidates[:, 1].min(),
                                        candidates[:, 2].max(), candidates[:, 3].max()])
                        width, height = box[2] - box[0], box[3] - box[1]
                        box += [-width * 0.2, -height * 0.08, width * 0.2, height * 0.08]
                        boxes[index, modality] = box.clip(0, 1)
                        confidence[index, modality] = max(scores)
                    else:
                        boxes[index, modality] = [0, 0, 1, 1]
    np.savez(ROOT / "data/metadata/person_boxes.npz", boxes=boxes, confidence=confidence)
    print("Detected clips per modality:", (confidence > 0).sum(0), flush=True)


def crop_clip(task):
    record, boxes, size = task
    images = np.zeros((16, 9, size, size), dtype=np.uint8)
    mask = np.zeros(3, dtype=np.float32)
    failures = []
    for modality_index, modality in enumerate(["IR", "Thermal", "Depth_Color"]):
        files = image_files(record, modality)
        if not files:
            continue
        box = boxes[1 if modality == "Thermal" else 0]
        selected = np.linspace(0, len(files) - 1, 16).round().astype(int)
        valid = {}
        for index in np.unique(selected):
            try:
                with Image.open(files[index]) as source:
                    image = enhance_infrared(source) if modality == "IR" else source.convert("RGB")
                    width, height = image.size
                    image = image.crop((int(box[0] * width), int(box[1] * height),
                                        max(int(box[0] * width) + 1, int(box[2] * width)),
                                        max(int(box[1] * height) + 1, int(box[3] * height))))
                    image = image.resize((size, size), Image.Resampling.BILINEAR)
                    valid[int(index)] = np.asarray(image).transpose(2, 0, 1)
            except (OSError, ValueError):
                failures.append(str(files[index].relative_to(ROOT)))
        if valid:
            mask[modality_index] = 1
            for position, index in enumerate(selected):
                images[position, modality_index * 3:(modality_index + 1) * 3] = valid[min(valid, key=lambda candidate: abs(candidate - index))]
    np.save(ROOT / f"data/processed/person_16_{size}/{record['index']:05d}.npy", images)
    return mask, failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["detect", "crop"], required=True)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--detector", type=Path)
    args = parser.parse_args()
    if args.stage == "detect":
        detect(args.detector)
    else:
        (ROOT / f"data/processed/person_16_{args.size}").mkdir(parents=True, exist_ok=True)
        records = get_records()
        boxes = np.load(ROOT / "data/metadata/person_boxes.npz")["boxes"]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            outputs = list(tqdm(pool.map(crop_clip, ((record, boxes[record["index"]], args.size) for record in records), chunksize=4),
                                total=len(records), desc="Person crops"))
        np.save(ROOT / "data/processed/person_mask.npy", np.stack([output[0] for output in outputs]))
        (ROOT / "data/metadata/person_crop_errors.json").write_text(json.dumps([error for output in outputs for error in output[1]], indent=2) + "\n")


if __name__ == "__main__":
    main()

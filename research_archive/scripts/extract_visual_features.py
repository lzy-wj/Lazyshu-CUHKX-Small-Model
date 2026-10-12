import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import ROOT, get_records


def enhance_infrared(image):
    values = np.asarray(image.convert("L"), dtype=np.float32)
    lower, upper = np.percentile(values, [1, 99])
    values = np.clip((values - lower) / max(upper - lower, 1) * 255, 0, 255).astype(np.uint8)
    enhanced = Image.fromarray(values)
    enhanced = Image.blend(enhanced, ImageOps.equalize(enhanced), 0.35)
    return enhanced.convert("RGB")


def motion_box(frames):
    small = np.stack([np.asarray(frame.convert("L").resize((80, 60)), dtype=np.float32) for frame in frames])
    motion = gaussian_filter(np.std(small, axis=0), 2)
    threshold = max(float(np.percentile(motion, 85)), 2.0)
    rows, columns = np.where(motion >= threshold)
    if len(rows) < 20:
        return (0.12, 0.08, 0.88, 0.92)
    left, right = np.percentile(columns, [2, 98]) / 80
    top, bottom = np.percentile(rows, [2, 98]) / 60
    center_x, center_y = (left + right) / 2, (top + bottom) / 2
    width, height = max(0.55, (right - left) * 1.3), max(0.65, (bottom - top) * 1.2)
    left = np.clip(center_x - width / 2, 0, max(0, 1 - width))
    top = np.clip(center_y - height / 2, 0, max(0, 1 - height))
    return float(left), float(top), min(1, left + width), min(1, top + height)


class VisualFrames(Dataset):
    def __init__(self, frames=8):
        self.records = get_records()
        self.frames = frames

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        images, mask, errors = [], [], []
        for modality in ["IR", "Thermal", "Depth_Color"]:
            folder = ROOT / record["modalities"].get(modality, "missing")
            files = sorted(path for path in folder.glob("*") if path.suffix.lower() in [".png", ".jpg", ".jpeg"]
                           and not path.name.startswith("."))
            selected = np.linspace(0, max(0, len(files) - 1), self.frames).round().astype(int)
            valid = {}
            for position in np.unique(selected):
                if not files:
                    break
                try:
                    with Image.open(files[position]) as image:
                        valid[int(position)] = enhance_infrared(image) if modality == "IR" else image.convert("RGB").copy()
                except (OSError, ValueError):
                    errors.append(str(files[position].relative_to(ROOT)))
            mask.append(float(bool(valid)))
            if valid:
                frames = [valid[min(valid, key=lambda candidate: abs(candidate - position))] for position in selected]
            else:
                frames = [Image.new("RGB", (640, 480))] * self.frames
            box = motion_box(frames)
            views = []
            for frame in frames:
                width, height = frame.size
                cropped = frame.crop((int(box[0] * width), int(box[1] * height),
                                      int(box[2] * width), int(box[3] * height)))
                for view in [frame, cropped]:
                    views.append(np.asarray(view.resize((224, 224), Image.Resampling.BILINEAR)).transpose(2, 0, 1))
            images.append(np.stack(views))
        return {"images": np.stack(images), "mask": np.array(mask, dtype=np.float32),
                "index": index, "errors": json.dumps(errors)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    os.environ.setdefault("TORCH_HOME", str(ROOT / ".tools/torch"))
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
    torch.set_num_threads(6)
    torch.backends.cudnn.benchmark = True
    backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    backbone.classifier = torch.nn.Identity()
    backbone = backbone.cuda().eval()
    dataset = VisualFrames(args.frames)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    outputs = {name: np.lib.format.open_memmap(ROOT / f"data/processed/{name}.npy", mode="w+", dtype=np.float32,
                                              shape=(len(dataset), args.frames, 2560))
               for name in ["ir_features", "thermal_features", "depth_features"]}
    masks = np.zeros((len(dataset), 3), dtype=np.float32)
    errors = []
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").reshape(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").reshape(1, 3, 1, 1)
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Frozen visual features"):
            images = batch["images"].flatten(0, 2).cuda(non_blocking=True).float() / 255
            images = (images - mean) / std
            chunks = []
            for chunk in images.split(128):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    chunks.append(backbone(chunk).float())
            values = torch.cat(chunks).reshape(len(batch["index"]), 3, args.frames, 2560).cpu().numpy()
            indices = batch["index"].numpy()
            for modality_index, output in enumerate(outputs.values()):
                output[indices] = values[:, modality_index]
            masks[indices] = batch["mask"].numpy()
            errors.extend(error for entry in batch["errors"] for error in json.loads(entry))
    for output in outputs.values():
        output.flush()
    np.save(ROOT / "data/processed/feature_mask.npy", masks)
    report = {"frames": args.frames, "backbone": "torchvision efficientnet_b0 IMAGENET1K_V1",
              "fit_on_target_data": False, "views": ["global", "motion_crop"], "image_size": 224,
              "errors": errors, "clips": len(dataset),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (ROOT / "data/metadata/visual_feature_extraction.json").write_text(json.dumps(report, indent=2) + "\n")
    print("DONE", len(dataset), "clips; bad frames:", len(errors), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional
from tqdm import trange


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-clips", type=int, default=12)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "data/processed/r2plus1d_yolo_all_visual_32_128",
    )
    parser.add_argument("--output", type=Path, default=HERE / "features")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.frames < 2 or args.batch_clips < 1:
        raise ValueError("Invalid extraction configuration")
    source_manifest = json.loads((args.source / "manifest.json").read_text())
    source_shape = tuple(source_manifest["shape"])
    if source_shape[2] != 7:
        raise ValueError("V134 requires RGB, Thermal, and IR source streams")
    source = np.memmap(
        args.source / "frames.uint8.memmap",
        dtype=np.uint8,
        mode="r",
        shape=source_shape,
    )
    positions = np.linspace(0, source_shape[1] - 1, args.frames).round().astype(int)

    torch_home = ROOT / ".tools/torch"
    torch.hub.set_dir(str(torch_home / "hub"))
    model = torch.hub.load(
        str(torch_home / "hub/facebookresearch_dinov2_main"),
        "dinov2_vits14",
        source="local",
        pretrained=True,
    ).eval().to(args.device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    args.output.mkdir(parents=True, exist_ok=True)
    shape = (source_shape[0], 3, args.frames, 768)
    feature_path = args.output / "dinov2_vits14_cls_patchmean.float16.memmap"
    features = np.memmap(feature_path, dtype=np.float16, mode="w+", shape=shape)
    visual_mask = np.zeros((source_shape[0], 3), dtype=np.float32)
    mean = torch.tensor((0.485, 0.456, 0.406), device=args.device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=args.device).view(1, 3, 1, 1)

    with torch.inference_mode():
        for start in trange(0, source_shape[0], args.batch_clips, desc="DINOv2"):
            stop = min(start + args.batch_clips, source_shape[0])
            values = torch.from_numpy(np.asarray(source[start:stop, positions]).copy()).to(
                args.device, non_blocking=True
            )
            streams = [values[:, :, :3], values[:, :, 3:6], values[:, :, 6:7].expand(-1, -1, 3, -1, -1)]
            for stream_index, stream in enumerate(streams):
                visual_mask[start:stop, stream_index] = (
                    stream.flatten(1).any(dim=1).float().cpu().numpy()
                )
                batch, steps, channels, height, width = stream.shape
                images = stream.reshape(batch * steps, channels, height, width).float().div_(255)
                images = functional.interpolate(
                    images, size=(224, 224), mode="bilinear", align_corners=False
                )
                images = (images - mean) / std
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model.forward_features(images)
                cls = output["x_norm_clstoken"].float()
                patch_mean = output["x_norm_patchtokens"].float().mean(dim=1)
                encoded = torch.cat([cls, patch_mean], dim=1).reshape(batch, steps, 768)
                features[start:stop, stream_index] = encoded.cpu().numpy().astype(np.float16)
    features.flush()
    np.save(args.output / "visual_mask.npy", visual_mask)

    weights_path = torch_home / "hub/checkpoints/dinov2_vits14_pretrain.pth"
    manifest = {
        "complete": True,
        "shape": list(shape),
        "dtype": "float16",
        "bytes": feature_path.stat().st_size,
        "frames": args.frames,
        "frame_positions": positions.tolist(),
        "source": str(args.source),
        "source_manifest_sha256": hashlib.sha256(
            (args.source / "manifest.json").read_bytes()
        ).hexdigest(),
        "weights": str(weights_path),
        "weights_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        "encoder_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "representation": "normalized class token concatenated with mean normalized patch token",
        "available_streams": visual_mask.sum(axis=0).astype(int).tolist(),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

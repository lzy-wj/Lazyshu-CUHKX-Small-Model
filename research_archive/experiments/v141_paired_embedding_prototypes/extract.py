#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_r2plus1d_checkpoint import build_model, make_loader, model_inputs
from src.data import get_indices, get_records
from src.r2plus1d_models import dequantize_state_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--scope", choices=("valid", "train"), default="valid")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=128)
    return parser.parse_args()


def checkpoint_state(checkpoint):
    for key in ("model_int4", "model_int7", "model_int6", "model_int8"):
        if key in checkpoint:
            return dequantize_state_dict(checkpoint[key])
    return checkpoint["model"]


class Recorder:
    def __init__(self, model):
        self.current = {}
        self.handles = [
            model.normalization.register_forward_hook(self._single("fused")),
            model.video_projection.register_forward_hook(self._multiple("visual")),
            model.skeleton.register_forward_hook(self._single("skeleton")),
            model.imu.register_forward_hook(self._single("imu")),
        ]

    def _single(self, name):
        def hook(_module, _inputs, output):
            self.current[name] = output.detach().float()

        return hook

    def _multiple(self, name):
        def hook(_module, _inputs, output):
            self.current.setdefault(name, []).append(output.detach().float())

        return hook

    def reset(self):
        self.current = {}

    def features(self):
        visual = self.current["visual"]
        if len(visual) != 3:
            raise RuntimeError(f"Expected three visual streams, received {len(visual)}")
        values = {
            "fused": self.current["fused"],
            "depth": visual[0].mean(dim=1),
            "thermal": visual[1].mean(dim=1),
            "ir": visual[2].mean(dim=1),
            "skeleton": self.current["skeleton"].mean(dim=1),
            "imu": self.current["imu"].mean(dim=1),
        }
        return {name: functional.normalize(value, dim=1) for name, value in values.items()}

    def close(self):
        for handle in self.handles:
            handle.remove()


@torch.no_grad()
def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"].copy()
    if int(config["fold"]) != args.fold:
        raise ValueError("Checkpoint fold does not match --fold")
    model = build_model(config, checkpoint_state(checkpoint))
    records = get_records()
    _, valid_indices = get_indices(records, args.fold)
    indices = (
        valid_indices
        if args.scope == "valid"
        else [record["index"] for record in records if record["split"] == "train"]
    )
    loader = make_loader(indices, args, config, config["mode"])
    recorder = Recorder(model)
    storage = {
        f"{name}_{view}": []
        for name in ("fused", "depth", "thermal", "ir", "skeleton", "imu")
        for view in ("center", "flip")
    }
    logits = {"center": [], "flip": []}
    ordered_indices = []
    labels = []
    for batch in loader:
        video, keyword = model_inputs(batch, config["mode"])
        for view, values in (("center", video), ("flip", torch.flip(video, dims=(-1,)))):
            recorder.reset()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output, _ = model(values, **keyword)
            logits[view].append(output.float().cpu().numpy())
            for name, feature in recorder.features().items():
                storage[f"{name}_{view}"].append(feature.cpu().numpy())
        ordered_indices.extend(batch["index"].numpy().tolist())
        labels.extend(batch["label"].numpy().tolist())
    recorder.close()
    payload = {
        name: np.concatenate(values).astype(np.float16)
        for name, values in storage.items()
    }
    payload.update(
        {
            "logits_center": np.concatenate(logits["center"]).astype(np.float16),
            "logits_flip": np.concatenate(logits["flip"]).astype(np.float16),
            "indices": np.asarray(ordered_indices, dtype=np.int64),
            "labels": np.asarray(labels, dtype=np.int64),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    report = {
        "fold": args.fold,
        "scope": args.scope,
        "rows": len(ordered_indices),
        "checkpoint": str(args.checkpoint),
        "output": str(args.output),
        "feature_dimensions": {
            name: int(payload[f"{name}_center"].shape[1])
            for name in ("fused", "depth", "thermal", "ir", "skeleton", "imu")
        },
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

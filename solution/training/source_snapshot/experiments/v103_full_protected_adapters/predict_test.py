#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.v97_protected_visual_adapters.model import build_from_source
from scripts.evaluate_r2plus1d_checkpoint import make_loader, predict
from src.data import get_records


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(args.adapter, map_location="cpu", weights_only=False)
    adapter_config = checkpoint["config"]
    location = adapter_config["location"]
    model, _, source_config = build_from_source(
        args.source,
        bottleneck=int(adapter_config["bottleneck"]),
        output_adapters=location == "dual",
    )
    result = model.load_state_dict(checkpoint["adapter"], strict=False)
    if result.unexpected_keys:
        raise ValueError(f"Unexpected adapter tensors: {result.unexpected_keys}")
    adapter_prefixes = ("base.video.intermediate_adapters.", "base.video.output_adapters.")
    missing_adapter = [name for name in result.missing_keys if name.startswith(adapter_prefixes)]
    if missing_adapter:
        raise ValueError(f"Missing adapter tensors: {missing_adapter}")
    model.cuda().eval()
    records = get_records()
    indices = [record["index"] for record in records if record["split"] == "test"]
    loader_args = argparse.Namespace(
        cache_dir=args.cache_dir,
        frames=32,
        image_size=160,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    loader = make_loader(indices, loader_args, source_config, "shared_multimodal")
    rows = predict(model, loader, "shared_multimodal", False, 1.0)
    ordered = {index: probabilities for index, _, probabilities, _ in rows}
    ordered_indices = np.asarray(indices, dtype=np.int64)
    probabilities = np.stack([ordered[int(index)] for index in ordered_indices])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, probabilities=probabilities, indices=ordered_indices)
    report = {
        "source": str(args.source),
        "adapter": str(args.adapter),
        "location": location,
        "rows": len(indices),
        "mean_confidence": float(probabilities.max(axis=1).mean()),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

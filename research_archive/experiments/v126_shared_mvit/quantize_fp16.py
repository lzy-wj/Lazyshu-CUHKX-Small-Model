#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    state = {
        name: value.half() if value.is_floating_point() else value
        for name, value in checkpoint["model"].items()
    }
    output = {
        "model": state,
        "config": checkpoint["config"],
        "epoch": checkpoint["epoch"],
        "metrics": checkpoint.get("metrics", {}),
        "storage": "FP16 parameters and buffers; inference graph restores FP32 modules",
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    report = {
        "input": str(args.input),
        "output": str(args.output),
        "input_bytes": args.input.stat().st_size,
        "output_bytes": args.output.stat().st_size,
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

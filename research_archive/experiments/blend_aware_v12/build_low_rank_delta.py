#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.r2plus1d_models import dequantize_state_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--ranks", type=int, nargs="+", default=(4, 8, 12, 16))
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def state(checkpoint):
    for key in ("model_int8", "model_int7", "model_int6", "model_int4"):
        if key in checkpoint:
            return dequantize_state_dict(checkpoint[key])
    return checkpoint["model"]


def quantize_rows(values):
    scale = (values.abs().amax(dim=1, keepdim=True) / 127).clamp_min(1e-12)
    integer = (values / scale).round().clamp(-127, 127).to(torch.int8)
    return integer.cpu(), scale.half().cpu()


def compact_adapter(entries, exact):
    integer_values = []
    scale_values = []
    metadata = []
    integer_offset = 0
    scale_offset = 0
    for name, left_integer, left_scale, right_integer, right_scale, shape in entries:
        left_integer = left_integer.flatten()
        right_integer = right_integer.flatten()
        left_scale = left_scale.flatten()
        right_scale = right_scale.flatten()
        metadata.append({
            "name": name,
            "shape": shape,
            "left_shape": tuple(left_integer.shape),
            "left_rows": left_scale.numel(),
            "left_integer_offset": integer_offset,
            "left_integer_count": left_integer.numel(),
            "left_scale_offset": scale_offset,
            "right_rows": right_scale.numel(),
            "right_integer_offset": integer_offset + left_integer.numel(),
            "right_integer_count": right_integer.numel(),
            "right_scale_offset": scale_offset + left_scale.numel(),
        })
        integer_values.extend((left_integer, right_integer))
        scale_values.extend((left_scale, right_scale))
        integer_offset += left_integer.numel() + right_integer.numel()
        scale_offset += left_scale.numel() + right_scale.numel()
    return {
        "format": "compact_low_rank_delta_v1",
        "entries": metadata,
        "integer": torch.cat(integer_values) if integer_values else torch.empty(0, dtype=torch.int8),
        "scale": torch.cat(scale_values) if scale_values else torch.empty(0, dtype=torch.float16),
        "exact": exact,
    }


def expand_adapter(adapter, base):
    output = {name: value.clone() for name, value in base.items()}
    integer = adapter["integer"]
    scale = adapter["scale"]
    for entry in adapter["entries"]:
        left_count = entry["left_integer_count"]
        right_count = entry["right_integer_count"]
        left_rows = entry["left_rows"]
        right_rows = entry["right_rows"]
        left = integer[
            entry["left_integer_offset"]:entry["left_integer_offset"] + left_count
        ].reshape(left_rows, -1).float()
        right = integer[
            entry["right_integer_offset"]:entry["right_integer_offset"] + right_count
        ].reshape(right_rows, -1).float()
        left_scale = scale[
            entry["left_scale_offset"]:entry["left_scale_offset"] + left_rows
        ].reshape(left_rows, 1).float()
        right_scale = scale[
            entry["right_scale_offset"]:entry["right_scale_offset"] + right_rows
        ].reshape(right_rows, 1).float()
        delta = (left * left_scale) @ (right * right_scale)
        output[entry["name"]] = output[entry["name"]] + delta.reshape(entry["shape"])
    for name, delta in adapter["exact"].items():
        output[name] = output[name] + delta.to(output[name].dtype) if delta.is_floating_point() else delta
    return output


def main():
    args = parse_args()
    ranks = sorted(set(args.ranks))
    if not ranks or ranks[0] < 1:
        raise ValueError("Ranks must be positive")
    device = torch.device(args.device)
    torch.manual_seed(42)
    base_checkpoint = torch.load(args.base, map_location="cpu", weights_only=False)
    target_checkpoint = torch.load(args.target, map_location="cpu", weights_only=False)
    base = state(base_checkpoint)
    target = state(target_checkpoint)
    if base.keys() != target.keys():
        raise ValueError("Base and target state dictionaries differ")
    decomposed = []
    exact = {}
    maximum_rank = ranks[-1]
    for position, (name, base_value) in enumerate(base.items(), 1):
        target_value = target[name]
        if base_value.shape != target_value.shape:
            raise ValueError(f"State shape mismatch: {name}")
        if not base_value.is_floating_point() or base_value.ndim < 2:
            exact[name] = (
                (target_value.float() - base_value.float()).half()
                if base_value.is_floating_point() else target_value.clone()
            )
            continue
        matrix = (target_value.float() - base_value.float()).reshape(base_value.shape[0], -1).to(device)
        rank = min(maximum_rank, *matrix.shape)
        if rank == min(matrix.shape):
            left, singular, right_h = torch.linalg.svd(matrix, full_matrices=False)
            left = left[:, :rank]
            singular = singular[:rank]
            right_h = right_h[:rank]
        else:
            left, singular, right = torch.svd_lowrank(
                matrix, q=min(rank + 4, *matrix.shape), niter=4
            )
            order = singular.argsort(descending=True)[:rank]
            left = left[:, order]
            singular = singular[order]
            right_h = right[:, order].T
        decomposed.append((name, (left * singular).cpu(), right_h.cpu(), tuple(base_value.shape)))
        del matrix, left, singular, right_h
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"decomposed {position}/{len(base)} {name}", flush=True)

    args.output_directory.mkdir(parents=True, exist_ok=True)
    reports = []
    for rank in ranks:
        entries = []
        for name, left, right, shape in decomposed:
            selected_rank = min(rank, left.shape[1])
            left_integer, left_scale = quantize_rows(left[:, :selected_rank])
            right_integer, right_scale = quantize_rows(right[:selected_rank])
            entries.append((
                name, left_integer, left_scale, right_integer, right_scale, shape,
            ))
        adapter = compact_adapter(entries, exact)
        adapter_path = args.output_directory / f"adapter_rank{rank:02d}.pt"
        torch.save(adapter, adapter_path)
        restored = expand_adapter(adapter, base)
        checkpoint = {
            "model": {
                name: value.half() if value.is_floating_point() else value
                for name, value in restored.items()
            },
            "config": target_checkpoint["config"],
            "epoch": target_checkpoint.get("epoch"),
            "metrics": target_checkpoint.get("metrics", {}),
            "low_rank_adapter": str(adapter_path),
            "low_rank_base": str(args.base),
        }
        reconstructed_path = args.output_directory / f"reconstructed_rank{rank:02d}.pt"
        torch.save(checkpoint, reconstructed_path)
        squared_error = 0
        squared_target = 0
        for name, value in target.items():
            if value.is_floating_point():
                difference = restored[name].float() - value.float()
                squared_error += difference.square().sum().item()
                squared_target += value.float().square().sum().item()
        report = {
            "rank": rank,
            "adapter": str(adapter_path),
            "adapter_bytes": adapter_path.stat().st_size,
            "reconstructed": str(reconstructed_path),
            "relative_state_error": (squared_error / squared_target) ** 0.5,
        }
        reports.append(report)
        print(json.dumps(report), flush=True)
    (args.output_directory / "report.json").write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()

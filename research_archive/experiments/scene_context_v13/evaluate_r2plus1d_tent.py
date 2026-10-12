#!/usr/bin/env python3
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_r2plus1d_checkpoint import adapt_batch_norm, build_model, make_loader, model_inputs, predict
from src.data import get_indices, get_records
from src.r2plus1d_models import dequantize_state_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--entropy-threshold", type=float, default=10.0)
    parser.add_argument("--adabn", choices=("none", "reset", "keep"), default="reset")
    parser.add_argument("--grouping", choices=("global", "user"), default="user")
    parser.add_argument("--tta-hflip", action="store_true")
    parser.add_argument("--tta-crop-scale", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def checkpoint_state(checkpoint):
    for key in ("model_int4", "model_int7", "model_int6", "model_int8"):
        if key in checkpoint:
            return dequantize_state_dict(checkpoint[key])
    return checkpoint["model"]


def configure_tent(model):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    parameters = []
    names = []
    for module_name, layer in model.named_modules():
        if not isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
            continue
        layer.train()
        if layer.weight is not None:
            layer.weight.requires_grad_(True)
            parameters.append(layer.weight)
            names.append(f"{module_name}.weight")
        if layer.bias is not None:
            layer.bias.requires_grad_(True)
            parameters.append(layer.bias)
            names.append(f"{module_name}.bias")
    if not parameters:
        raise RuntimeError("TENT requires affine batch-normalization parameters")
    return parameters, names


def adapt_tent(model, loader, mode, args):
    parameters, names = configure_tent(model)
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    history = []
    for adaptation_pass in range(args.passes):
        entropy_sum = 0.0
        selected_rows = 0
        total_rows = 0
        for batch in loader:
            video, keyword = model_inputs(batch, mode)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(video, **keyword)
            probabilities = logits.float().softmax(1)
            entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(1)
            selected = entropy < args.entropy_threshold
            if not selected.any():
                continue
            loss = entropy[selected].mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            entropy_sum += float(entropy[selected].detach().sum())
            selected_rows += int(selected.sum())
            total_rows += len(entropy)
        history.append(
            {
                "pass": adaptation_pass + 1,
                "mean_selected_entropy": entropy_sum / max(selected_rows, 1),
                "selected_rows": selected_rows,
                "total_rows": total_rows,
            }
        )
    model.eval()
    return names, history


def main():
    args = parse_args()
    if args.lr <= 0 or args.passes < 1:
        raise ValueError("TENT lr and passes must be positive")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    mode = config["mode"]
    state = checkpoint_state(checkpoint)
    records = get_records()
    _, indices = get_indices(records, args.fold)
    groups = [indices]
    if args.grouping == "user":
        groups = [
            [index for index in indices if records[index]["user"] == user]
            for user in sorted({records[index]["user"] for index in indices})
        ]
    rows = []
    adaptation = []
    parameter_names = []
    for group in groups:
        model = build_model(config, copy.deepcopy(state))
        loader = make_loader(group, args, config, mode)
        if args.adabn != "none":
            adapt_batch_norm(model, loader, mode, passes=1, reset=args.adabn == "reset")
        parameter_names, history = adapt_tent(model, loader, mode, args)
        rows.extend(predict(model, loader, mode, args.tta_hflip, args.tta_crop_scale))
        adaptation.append(
            {
                "user": records[group[0]]["user"],
                "rows": len(group),
                "history": history,
            }
        )
        del model
        torch.cuda.empty_cache()
    ordered = {index: (label, values) for index, label, values, _ in rows}
    ordered_indices = np.asarray(indices, dtype=np.int64)
    labels = np.asarray([ordered[int(index)][0] for index in ordered_indices])
    probabilities = np.stack([ordered[int(index)][1] for index in ordered_indices])
    report = {
        "accuracy": float((probabilities.argmax(1) == labels).mean()),
        "fold": args.fold,
        "lr": args.lr,
        "passes": args.passes,
        "entropy_threshold": args.entropy_threshold,
        "adabn": args.adabn,
        "grouping": args.grouping,
        "adapted_parameters": int(sum(state[name].numel() for name in parameter_names)),
        "adaptation": adaptation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, probabilities=probabilities, labels=labels, indices=ordered_indices)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

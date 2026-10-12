#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_r2plus1d_checkpoint import build_model, make_loader, predict
from scripts.evaluate_repeat_decoder import decode, transition_matrix
from src.data import FOLDS, get_indices, get_records


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right-weight", type=float, default=0.75)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--selection-objective", choices=("raw", "decoded", "nll"), default="raw")
    parser.add_argument("--epochs", type=int, nargs="+")
    parser.add_argument("--allow-right-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def state(checkpoint):
    if "model" not in checkpoint:
        raise ValueError("Epoch soup evaluation requires FP16 model states")
    return checkpoint["model"]


def interpolate(left, right, right_weight, allow_right_only=False):
    if right_weight == 1:
        return {name: value.clone() for name, value in right.items()}
    if left.keys() != right.keys() and not (allow_right_only and left.keys() <= right.keys()):
        raise ValueError("Soup checkpoint state dictionaries differ")
    mixed = {}
    for name in right:
        if name not in left:
            mixed[name] = right[name].clone()
        elif left[name].shape != right[name].shape:
            raise ValueError(f"Soup tensor shape mismatch: {name}")
        else:
            mixed[name] = (
                left[name].float().lerp(right[name].float(), right_weight).half()
                if left[name].is_floating_point() else right[name].clone()
            )
    return mixed


def load(path):
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key] for key in values.files}


def align(reference, candidate):
    positions = {int(index): position for position, index in enumerate(candidate["indices"])}
    order = np.asarray([positions[int(index)] for index in reference["indices"]])
    if "labels" in candidate and not np.array_equal(reference["labels"], candidate["labels"][order]):
        raise ValueError("Prediction labels do not align")
    return candidate["probabilities"][order].astype(np.float64)


def temperature_scale(probabilities, temperature):
    logits = np.log(np.clip(probabilities, 1e-12, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Epoch soup evaluation requires CUDA")
    if not 0 <= args.right_weight <= 1:
        raise ValueError("Right soup weight must be in [0, 1]")
    epoch_paths = sorted((args.run_directory / "epochs").glob("epoch_*.pt"))
    if args.epochs is not None:
        selected_epochs = set(args.epochs)
        epoch_paths = [path for path in epoch_paths if int(path.stem.removeprefix("epoch_")) in selected_epochs]
        missing = selected_epochs - {int(path.stem.removeprefix("epoch_")) for path in epoch_paths}
        if missing:
            raise FileNotFoundError(f"Missing requested epoch checkpoints: {sorted(missing)}")
    if not epoch_paths:
        raise FileNotFoundError(f"No epoch checkpoints in {args.run_directory / 'epochs'}")
    left_checkpoint = torch.load(args.left, map_location="cpu", weights_only=False)
    left_state = state(left_checkpoint)
    first = torch.load(epoch_paths[0], map_location="cpu", weights_only=False)
    config = first["config"]
    if int(config["fold"]) != args.fold:
        raise ValueError("Run checkpoint fold mismatch")
    records = get_records()
    _, indices = get_indices(records, args.fold)
    loader_args = SimpleNamespace(
        cache_dir=args.cache_dir,
        frames=args.frames,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    loader = make_loader(indices, loader_args, config, config["mode"])
    base = load(ROOT / f"outputs/study_r2plus1d/lightweight_base_int8/base_fold{args.fold}.npz")
    auxiliary = align(
        base,
        load(ROOT / f"outputs/study_x3dl_soup/selected_cropblend50_fold{args.fold}.npz"),
    )
    transitions = [
        transition_matrix(records, set(FOLDS[args.fold]), environment=environment)
        for environment in (0, 1)
    ]
    output_predictions = args.output.parent / f"{args.output.stem}_predictions"
    output_predictions.mkdir(parents=True, exist_ok=True)
    results = []
    for path in epoch_paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        mixed = interpolate(
            left_state, state(checkpoint), args.right_weight, allow_right_only=args.allow_right_only
        )
        model = build_model(config, mixed)
        rows = predict(model, loader, config["mode"], tta_hflip=True, crop_scale=0.75)
        del model, mixed
        torch.cuda.empty_cache()
        predicted = {
            int(index): (int(label), probabilities)
            for index, label, probabilities, _ in rows
        }
        candidate_file = {
            "indices": np.asarray(indices, dtype=np.int64),
            "labels": np.asarray([predicted[int(index)][0] for index in indices], dtype=np.int64),
            "probabilities": np.stack([predicted[int(index)][1] for index in indices]),
        }
        candidate = align(base, candidate_file)
        probabilities = 0.4 * base["probabilities"] + 0.6 * temperature_scale(candidate, 2.0)
        gated = probabilities.max(axis=1) < 0.15
        probabilities[gated] = 0.9 * probabilities[gated] + 0.1 * auxiliary[gated]
        labels = base["labels"]
        decoded = decode(
            probabilities,
            base["indices"],
            records,
            maximum_gap=120,
            minimum_similarity=0.7,
            enforce_unique=True,
            transition_strength=0.4,
            environment_transitions=transitions,
            repeat_pool_power=2,
        )
        metrics = {
            "epoch": int(checkpoint["epoch"]),
            "checkpoint": str(path),
            "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "standalone_accuracy": float((candidate.argmax(axis=1) == labels).mean()),
            "raw_accuracy": float((probabilities.argmax(axis=1) == labels).mean()),
            "nll": float(-np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1)).mean()),
            "decoded_accuracy": float(decoded["accuracy"]),
            "decoded_correct": round(float(decoded["accuracy"]) * len(labels)),
            "gated_rows": int(gated.sum()),
        }
        results.append(metrics)
        np.savez(
            output_predictions / f"epoch_{metrics['epoch']:02d}.npz",
            probabilities=candidate.astype(np.float32),
            labels=labels,
            indices=base["indices"],
        )
        print(json.dumps(metrics), flush=True)
    if args.selection_objective == "raw":
        selected = max(results, key=lambda item: (item["raw_accuracy"], -item["nll"], item["decoded_accuracy"]))
    elif args.selection_objective == "decoded":
        selected = max(results, key=lambda item: (item["decoded_accuracy"], item["raw_accuracy"], -item["nll"]))
    else:
        selected = min(results, key=lambda item: (item["nll"], -item["raw_accuracy"], -item["decoded_accuracy"]))
    report = {
        "fold": args.fold,
        "selection_objective": args.selection_objective,
        "right_weight": args.right_weight,
        "tta": "original, horizontal flip, center 75% crop, crop plus horizontal flip",
        "left": str(args.left),
        "selected": selected,
        "v9_reference": {
            "raw_accuracy": (0.7101226993865031, 0.7914893617021277)[args.fold],
            "decoded_correct": (501, 594)[args.fold],
        },
        "epochs": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "epochs"}, indent=2))


if __name__ == "__main__":
    main()

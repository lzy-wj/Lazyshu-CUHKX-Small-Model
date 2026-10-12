#!/usr/bin/env python3
import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.v104_domain_gated_adapters.evaluate import fold_probabilities
from experiments.v109_crossfit_round_transformer.model import (
    CrossFittedRoundTransformer,
    normalized_log_probabilities,
)
from experiments.v84_residual_invariant_skeleton.evaluate import domains
from scripts.evaluate_repeat_decoder import rounds
from src.data import get_records


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--swanlab-mode", choices=("cloud", "local", "disabled"), default="cloud")
    return parser.parse_args()


def v104_probabilities(fold, records):
    reference, v89, protected, old_target, refreshed_target = fold_probabilities(fold)
    domain = domains(reference["indices"], fold, records)
    candidate = v89.copy()
    candidate[domain == 1] = (
        v89[domain == 1]
        + protected[domain == 1]
        + old_target[domain == 1]
        + refreshed_target[domain == 1]
    ) / 4
    candidate[domain == 2] = 0.5 * (
        v89[domain == 2] + protected[domain == 2]
    )
    return reference, v89, candidate, domain


def make_batch(reference, v89, v104, domain, records):
    indices = reference["indices"].astype(np.int64)
    positions = {int(index): position for position, index in enumerate(indices)}
    round_values = rounds(indices, records)
    covered = {index for item in round_values for index in item["rows"]}
    round_values.extend(
        {"rows": [int(index)]}
        for index in indices
        if int(index) not in covered
    )
    maximum_length = max(len(item["rows"]) for item in round_values)
    if maximum_length > 16:
        raise ValueError("Round length exceeds V109 positional capacity")
    v89_logs = normalized_log_probabilities(v89)
    v104_logs = normalized_log_probabilities(v104)
    features = np.zeros((len(round_values), maximum_length, 80), dtype=np.float32)
    anchors = np.zeros((len(round_values), maximum_length, 40), dtype=np.float32)
    labels = np.full((len(round_values), maximum_length), -100, dtype=np.int64)
    padding = np.ones((len(round_values), maximum_length), dtype=bool)
    domain_values = np.zeros(len(round_values), dtype=np.int64)
    lengths = np.zeros(len(round_values), dtype=np.int64)
    output_positions = []
    for round_position, item in enumerate(round_values):
        row_positions = np.asarray([positions[int(index)] for index in item["rows"]])
        length = len(row_positions)
        row_domains = np.unique(domain[row_positions])
        if len(row_domains) != 1:
            raise ValueError("A V109 round crosses domains")
        features[round_position, :length] = np.concatenate(
            [v89_logs[row_positions], v104_logs[row_positions]], axis=1
        )
        anchors[round_position, :length] = v104_logs[row_positions]
        labels[round_position, :length] = reference["labels"][row_positions]
        padding[round_position, :length] = False
        domain_values[round_position] = row_domains[0]
        lengths[round_position] = length
        output_positions.append(row_positions)
    return {
        "features": torch.from_numpy(features),
        "anchors": torch.from_numpy(anchors),
        "labels": torch.from_numpy(labels),
        "padding": torch.from_numpy(padding),
        "domains": torch.from_numpy(domain_values),
        "lengths": torch.from_numpy(lengths),
        "output_positions": output_positions,
        "rows": len(indices),
    }


def soft_uniqueness(probabilities, padding):
    valid = (~padding).to(probabilities.dtype)
    values = probabilities * valid.unsqueeze(2)
    overlaps = torch.bmm(values, values.transpose(1, 2))
    pair_mask = valid.unsqueeze(2) * valid.unsqueeze(1)
    diagonal = torch.eye(overlaps.shape[1], device=overlaps.device).unsqueeze(0)
    pair_mask = pair_mask * (1 - diagonal)
    return (overlaps * pair_mask).sum() / pair_mask.sum().clamp_min(1)


@torch.no_grad()
def predict(model, batch):
    model.eval()
    logits = model(
        batch["features"],
        batch["anchors"],
        batch["padding"],
        batch["domains"],
        batch["lengths"],
    )
    probabilities = logits.softmax(dim=2).numpy()
    output = np.empty((batch["rows"], 40), dtype=np.float32)
    for round_position, positions in enumerate(batch["output_positions"]):
        output[positions] = probabilities[round_position, : len(positions)]
    return output


def main():
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("Steps must be positive")
    torch.manual_seed(args.seed + args.fold)
    np.random.seed(args.seed + args.fold)
    torch.set_num_threads(min(torch.get_num_threads(), 16))
    records = get_records()
    source_fold = 1 - args.fold
    source_reference, source_v89, source_v104, source_domain = v104_probabilities(
        source_fold, records
    )
    target_reference, target_v89, target_v104, target_domain = v104_probabilities(
        args.fold, records
    )
    train_batch = make_batch(
        source_reference, source_v89, source_v104, source_domain, records
    )
    target_batch = make_batch(
        target_reference, target_v89, target_v104, target_domain, records
    )
    model = CrossFittedRoundTransformer()
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.02)
    valid = ~train_batch["padding"]
    anchor_probabilities = train_batch["anchors"].exp()
    name = f"v109_crossfit_round_transformer_f{args.fold}"
    destination = HERE / "outputs" / name
    destination.mkdir(parents=True, exist_ok=False)
    config = {
        **vars(args),
        "name": name,
        "source_fold": source_fold,
        "source_rows": train_batch["rows"],
        "target_rows": target_batch["rows"],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "architecture": "two-layer bidirectional Transformer residual over cross-fitted round logits",
        "training_protocol": "fixed endpoint on opposite-fold OOF probabilities and legal source labels",
        "model_sha256": hashlib.sha256((HERE / "model.py").read_bytes()).hexdigest(),
    }
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    import swanlab

    run = swanlab.init(
        project="cuhkx-small-multimodal",
        name=name,
        config=config,
        public=False,
        mode=args.swanlab_mode,
        log_dir=str(HERE / "swanlog"),
        description="Leakage-free cross-fitted acquisition-round Transformer.",
    )
    started = time.monotonic()
    history = []
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        noise_scale = 0.08 * (1 - 0.5 * step / args.steps)
        features = train_batch["features"] + torch.randn_like(
            train_batch["features"]
        ) * noise_scale
        logits = model(
            features,
            train_batch["anchors"],
            train_batch["padding"],
            train_batch["domains"],
            train_batch["lengths"],
        )
        classification = functional.cross_entropy(
            logits.reshape(-1, 40),
            train_batch["labels"].reshape(-1),
            ignore_index=-100,
            label_smoothing=0.03,
        )
        retention = functional.kl_div(
            functional.log_softmax(logits[valid], dim=1),
            anchor_probabilities[valid],
            reduction="batchmean",
        )
        uniqueness = soft_uniqueness(logits.softmax(dim=2), train_batch["padding"])
        loss = classification + 0.5 * retention + 0.05 * uniqueness
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        with torch.no_grad():
            decay = min(0.995, 1 - 1 / step)
            for target, source in zip(ema.parameters(), model.parameters()):
                target.lerp_(source, 1 - decay)
        if step == 1 or step % 50 == 0 or step == args.steps:
            with torch.no_grad():
                train_correct = int(
                    (logits.argmax(2)[valid] == train_batch["labels"][valid]).sum()
                )
            values = {
                "step": step,
                "loss": float(loss.detach()),
                "classification": float(classification.detach()),
                "retention": float(retention.detach()),
                "uniqueness": float(uniqueness.detach()),
                "source_correct": train_correct,
                "source_rows": train_batch["rows"],
            }
            history.append(values)
            swanlab.log({key: value for key, value in values.items() if key != "step"}, step=step)
            print(json.dumps(values), flush=True)

    probabilities = predict(ema, target_batch)
    state = {
        "format": "v109_crossfit_round_transformer_v1",
        "state_dict": {
            name: value.detach().half() if value.is_floating_point() else value.detach()
            for name, value in ema.state_dict().items()
        },
        "config": config,
    }
    torch.save(state, destination / "model.pt")
    np.savez(
        destination / "validation.npz",
        probabilities=probabilities,
        labels=target_reference["labels"],
        indices=target_reference["indices"],
    )
    (destination / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    metrics = {
        "rows": len(target_reference["labels"]),
        "correct": int(
            (probabilities.argmax(1) == target_reference["labels"]).sum()
        ),
        "accuracy": float(
            (probabilities.argmax(1) == target_reference["labels"]).mean()
        ),
        "elapsed_seconds": time.monotonic() - started,
    }
    try:
        metrics["swanlab_url"] = run.url
    except ValueError:
        metrics["swanlab_url"] = None
    (destination / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    swanlab.finish()
    print("DONE " + json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()

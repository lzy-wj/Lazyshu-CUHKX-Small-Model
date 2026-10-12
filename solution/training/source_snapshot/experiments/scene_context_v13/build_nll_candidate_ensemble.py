#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.blend_aware_v12.audit_existing import align, load
from experiments.scene_context_v13.research_sequence_decoder import selected_probabilities


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--patterns", nargs="+", required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args()


def simplex(values):
    values = np.exp(values - values.max())
    return values / values.sum()


def main():
    args = parse_args()
    if len(args.names) != len(args.patterns) or len(set(args.names)) != len(args.names):
        raise ValueError("Candidate names and patterns must be aligned and uniquely named")
    folds = []
    for fold in (0, 1):
        base, indices, labels = selected_probabilities(fold)
        reference = {"probabilities": base, "indices": indices, "labels": labels}
        candidates = np.stack(
            [
                align(reference, load(Path(pattern.format(fold=fold))))
                for pattern in args.patterns
            ]
        )
        folds.append((indices, labels, candidates))

    fold0_labels = folds[0][1]
    fold0_candidates = folds[0][2]

    def objective(values):
        probabilities = np.tensordot(simplex(values), fold0_candidates, axes=1)
        return float(
            -np.log(
                probabilities[np.arange(len(fold0_labels)), fold0_labels].clip(1e-8)
            ).mean()
        )

    fit = minimize(objective, np.zeros(len(args.names)), method="BFGS")
    weights = simplex(fit.x)
    report = {
        "protocol": "Convex weights minimize fold0 NLL; fold1 is evaluated once blind.",
        "optimization_success": bool(fit.success),
        "weights": dict(zip(args.names, weights.tolist())),
        "folds": [],
    }
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for fold, (indices, labels, candidates) in enumerate(folds):
        probabilities = np.tensordot(weights, candidates, axes=1)
        correct = int((probabilities.argmax(axis=1) == labels).sum())
        nll = float(
            -np.log(probabilities[np.arange(len(labels)), labels].clip(1e-8)).mean()
        )
        report["folds"].append(
            {"fold": fold, "rows": len(labels), "correct": correct, "nll": nll}
        )
        np.savez_compressed(
            Path(f"{args.output_prefix}_f{fold}.npz"),
            indices=indices,
            labels=labels,
            probabilities=probabilities,
        )
    Path(f"{args.output_prefix}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
